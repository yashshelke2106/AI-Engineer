"""
Raw dataset ingestion.

Design goal: the caller supplies nothing but a file path. No column names,
no target hint, no problem-type hint, no separator. Everything about the
file's shape is *discovered*, not configured. This module's job stops at
"I have a clean-ish pandas DataFrame and I know a few objective facts about
how I obtained it" — it does not clean or interpret the data; that is the
profiler's and cleaner's job.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class IngestionReport:
    source_path: str
    file_format: str
    detected_encoding: str
    detected_delimiter: str | None
    n_rows: int
    n_cols: int
    memory_mb: float
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "file_format": self.file_format,
            "detected_encoding": self.detected_encoding,
            "detected_delimiter": self.detected_delimiter,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_mb": round(self.memory_mb, 3),
            "warnings": self.warnings,
            "notes": self.notes,
        }


_ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def _sniff_delimiter(sample: str) -> str | None:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # Fall back to the most frequent plausible delimiter by raw count.
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else None


def _read_text_sample(path: Path, encoding: str, n_bytes: int = 65536) -> str:
    with open(path, "r", encoding=encoding, errors="strict") as f:
        return f.read(n_bytes)


def _detect_encoding_and_sample(path: Path) -> tuple[str, str]:
    last_err: Exception | None = None
    for enc in _ENCODINGS_TO_TRY:
        try:
            return enc, _read_text_sample(path, enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise ValueError(f"Could not decode {path} with any of {_ENCODINGS_TO_TRY}") from last_err


def load_raw_dataset(path: str | Path) -> tuple[pd.DataFrame, IngestionReport]:
    """
    Load a dataset with zero prior knowledge of its schema.

    Supports: .csv/.tsv/.txt (delimiter sniffed), .parquet, .json / .jsonl,
    .xlsx/.xls. Raises ValueError for anything else rather than guessing
    wrong silently.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    warnings: list[str] = []
    notes: list[str] = []
    suffix = path.suffix.lower()

    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        report = IngestionReport(
            source_path=str(path), file_format="parquet",
            detected_encoding="n/a", detected_delimiter=None,
            n_rows=len(df), n_cols=df.shape[1],
            memory_mb=df.memory_usage(deep=True).sum() / 1e6,
        )
        return df, report

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        report = IngestionReport(
            source_path=str(path), file_format="excel",
            detected_encoding="n/a", detected_delimiter=None,
            n_rows=len(df), n_cols=df.shape[1],
            memory_mb=df.memory_usage(deep=True).sum() / 1e6,
        )
        return df, report

    if suffix in (".json", ".jsonl", ".ndjson"):
        encoding, sample = _detect_encoding_and_sample(path)
        # Try JSON Lines first (one record per line), then a single JSON blob.
        is_jsonl = suffix in (".jsonl", ".ndjson")
        try:
            if is_jsonl:
                df = pd.read_json(path, lines=True, encoding=encoding)
            else:
                # Peek at the first non-whitespace char to decide array-of-records
                # vs. lines-of-records vs. a nested dict that needs normalizing.
                head = sample.lstrip()
                if head.startswith("["):
                    df = pd.read_json(path, encoding=encoding)
                elif head.startswith("{"):
                    with open(path, "r", encoding=encoding) as f:
                        obj = json.load(f)
                    if isinstance(obj, dict):
                        # Could be {"data": [...]} or a single flat record.
                        list_fields = [k for k, v in obj.items() if isinstance(v, list)]
                        if list_fields:
                            key = max(list_fields, key=lambda k: len(obj[k]))
                            df = pd.json_normalize(obj[key])
                            notes.append(f"Extracted records from top-level key '{key}'.")
                        else:
                            df = pd.json_normalize([obj])
                            warnings.append("JSON root was a single flat object; treated as a 1-row table.")
                    else:
                        df = pd.json_normalize(obj)
                else:
                    df = pd.read_json(path, lines=True, encoding=encoding)
                    notes.append("Fell back to JSON-Lines parsing.")
        except ValueError:
            df = pd.read_json(path, lines=True, encoding=encoding)
            notes.append("Fell back to JSON-Lines parsing after standard JSON parse failed.")

        report = IngestionReport(
            source_path=str(path), file_format="json",
            detected_encoding=encoding, detected_delimiter=None,
            n_rows=len(df), n_cols=df.shape[1],
            memory_mb=df.memory_usage(deep=True).sum() / 1e6,
            warnings=warnings, notes=notes,
        )
        return df, report

    # Default: treat as delimited text (.csv, .tsv, .txt, or unknown).
    encoding, sample = _detect_encoding_and_sample(path)
    delimiter = _sniff_delimiter(sample)
    if delimiter is None:
        delimiter = ","
        warnings.append("Could not confidently sniff a delimiter; defaulted to ','.")

    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = True
        notes.append("Header presence could not be sniffed; assumed a header row exists.")

    df = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        header=0 if has_header else None,
        engine="python",
        on_bad_lines="warn",
    )
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        warnings.append("No header row detected; synthetic column names assigned (col_0, col_1, ...).")

    # Drop fully-empty unnamed columns pandas sometimes creates from trailing delimiters.
    unnamed_all_na = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]
    if unnamed_all_na:
        df = df.drop(columns=unnamed_all_na)
        notes.append(f"Dropped {len(unnamed_all_na)} fully-empty trailing column(s).")

    report = IngestionReport(
        source_path=str(path), file_format="delimited_text",
        detected_encoding=encoding, detected_delimiter=delimiter,
        n_rows=len(df), n_cols=df.shape[1],
        memory_mb=df.memory_usage(deep=True).sum() / 1e6,
        warnings=warnings, notes=notes,
    )
    return df, report
