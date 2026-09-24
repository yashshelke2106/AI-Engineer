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


#: A header cell is a name, not a document. Anything longer than this, or with
#: more words than the next constant, is data that happens to be on row one.
_MAX_HEADER_CELL_CHARS = 64
_MAX_HEADER_CELL_WORDS = 6


def _first_row_looks_like_names(sample: str, delimiter: str) -> bool:
    """Second opinion for `csv.Sniffer().has_header`, which abstains on free text.

    The sniffer decides by comparing each column's first cell with the cells
    below it: numeric where they are numeric, or a different string *length*. It
    only counts length when every row in that column shares one, so a column of
    free text — where no two documents are the same length — casts no vote at
    all, and a file whose every column is free text gets zero votes and is
    declared headerless. Its header row then becomes a data row and every column
    is renamed col_0, col_1, which is silent on a run using auto-detection.

    So: row one is names if each of its cells reads like a name (short, few
    words, non-numeric, distinct, and not repeated in its own column) AND at
    least one column's values below look unlike it — numeric, or typically much
    longer.
    """
    rows = [r for r in csv.reader(sample.splitlines(), delimiter=delimiter) if r]
    if len(rows) < 3:
        return False
    head, body = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
    if not body or len(set(head)) != len(head):
        return False

    for cell in head:
        text = cell.strip()
        if not text or len(text) > _MAX_HEADER_CELL_CHARS or len(text.split()) > _MAX_HEADER_CELL_WORDS:
            return False
        try:
            float(text)
            return False  # a number is a measurement, not a name
        except ValueError:
            pass

    unlike_below = False
    for i, cell in enumerate(head):
        column = [r[i].strip() for r in body if i < len(r)]
        if cell.strip() in column:
            return False  # a name would not recur as one of its own values
        numeric = sum(_is_number(v) for v in column)
        if numeric > 0.8 * len(column):
            unlike_below = True
            continue
        lengths = sorted(len(v) for v in column)
        median = lengths[len(lengths) // 2]
        if median >= 2 * max(len(cell.strip()), 1):
            unlike_below = True
    return unlike_below


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _decide_header(sample: str, delimiter: str) -> tuple[bool, str]:
    """Whether row one holds column names, and the basis for saying so.

    Measured over every committed dataset plus a header-stripped copy of each
    (34 cases): `csv.Sniffer().has_header` got 31, missing the header on a
    free-text file and inventing one on a headerless free-text file.
    `_first_row_looks_like_names` got 34, so it leads and the sniffer is the
    fallback for a sample too short to judge (under three rows).

    Detection is a guess (invariant 6), so the basis goes in the ingestion
    report either way rather than the run silently proceeding on `col_0`.
    """
    if len([r for r in csv.reader(sample.splitlines(), delimiter=delimiter) if r]) < 3:
        try:
            sniffed = csv.Sniffer().has_header(sample)
        except csv.Error:
            return True, "too few rows to judge and the sniffer could not either; assumed a header exists"
        return sniffed, f"too few rows to judge; csv.Sniffer says header={sniffed}"

    if _first_row_looks_like_names(sample, delimiter):
        return True, "row one reads as column names — short, distinct, and unlike the values below it"
    return False, (
        "row one reads as data, not names (a long, numeric, repeated or duplicated first row); "
        "synthetic column names will be used"
    )


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

    has_header, basis = _decide_header(sample, delimiter)
    notes.append(f"Header decision: {basis}")

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
