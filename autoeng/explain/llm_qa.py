"""
T2-5 — an LLM front-end over the grounded lookups in `qa.py`.

The keyword router in `qa.py` answers from logged artifacts and nothing else,
but only questions phrased the way it expects. This puts Claude in front of the
same lookups as tools, so open-ended phrasing works — and then refuses to trust
the result until it has been checked.

The check is the point. A fluent answer is easy; a fluent answer with a wrong
number in it is the failure this project exists to prevent, and it reads exactly
like a right one. So every number in the model's answer must be one the tools
actually returned (to the precision the answer states it, or as a percentage of
one). An answer carrying a number no tool produced is not shown: the question
falls back to the keyword router, and the reply names the numbers that could not
be verified. Counts and ordinals up to ten are exempt — "the top 3 features" —
which is the only allowance. A number that appears only in the question does not
count either, or "did it score 0.99?" answered "yes, 0.99" would verify itself.

Everything degrades to the keyword router rather than failing: no SDK, no
credentials, an API error, a refusal, a runaway tool loop. The LLM is an
improvement to phrasing, never a dependency of `ask`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from autoeng.explain.qa import LOOKUPS, RunRecord, answer_question

DEFAULT_MODEL = "claude-opus-5"
MAX_TOOL_TURNS = 6
MAX_TOKENS = 4096
#: Server-side refusal fallback: a declined request is re-run on Anthropic's
#: recommended fallback model inside the same call instead of coming back empty.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Counts and ordinals ("the top 3") need not appear in a tool result.
STRUCTURAL_INTEGER_MAX = 10

SYSTEM_PROMPT = """\
You answer questions about a single AutoML run, using only the tools provided. Each tool \
returns facts logged by that run: its leaderboard, the selected model and why, leakage \
findings, how the problem type and target were detected, cleaning, tuning, feature \
importances, the decision threshold, and any champion-challenger decision.

Call the tools you need before answering. State numbers exactly as the tools return \
them (rounding to fewer decimal places is fine). If the tools do not contain what the \
question asks, say that plainly; do not estimate, infer a number, or describe what \
typically happens in other runs. Your answer is checked: any number that does not \
appear in a tool result causes it to be discarded. Keep answers short and specific."""


def _tool(name: str, description: str, properties: dict | None = None) -> dict:
    properties = properties or {}
    return {
        "name": name, "description": description, "strict": True,
        "input_schema": {"type": "object", "properties": properties,
                         "required": list(properties), "additionalProperties": False},
    }


_TOP_N = {"top_n": {"type": "integer", "description": "How many entries to return; use 10 unless asked."}}
TOOLS = [
    _tool("leaderboard", "Fully cross-validated candidates ranked by the primary metric, plus the "
          "ones that failed, were skipped or were screened out.", _TOP_N),
    _tool("candidate", "One candidate model's status, metrics, error if it failed, and its gap to "
          "the winner. Use for 'why was X rejected' or 'how did X do'.",
          {"model": {"type": "string", "description": "Model name as it appears on the leaderboard."}}),
    _tool("winner", "The selected model, its score, the runner-up, the margin, whether the margin "
          "is within cross-validation noise, and the tuning improvement."),
    _tool("leakage", "Data-leakage flags raised before and after training."),
    _tool("problem_detection", "Detected problem type and target, the reasoning, the confidence, "
          "and the alternative hypotheses considered."),
    _tool("cleaning", "Structural cleaning actions taken before splitting."),
    _tool("tuning", "Hyperparameter-tuning results per tuned model."),
    _tool("feature_importances", "The winner's most important features and the method used.", _TOP_N),
    _tool("operating_point", "The binary decision threshold, the objective it was chosen for, its "
          "out-of-fold precision/recall/F1, and probability calibration."),
    _tool("promotion", "The champion-challenger gate decision, if this run was a gated retrain."),
]


@dataclass
class LLMAnswer:
    text: str
    source: str  # "llm" or "keyword"
    grounded: bool
    unsupported_numbers: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None


# ---------------------------------------------------------------------------
# The grounding check.
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(_THOUSANDS.sub("", text))


def _decimals(token: str) -> int:
    body = token.rstrip("%")
    return len(body.split(".", 1)[1]) if "." in body else 0


def unsupported_numbers(answer: str, evidence: list[str]) -> list[str]:
    """Numbers in `answer` that no evidence string supports, at the answer's precision."""
    known = []
    for text in evidence:
        for token in _numbers(text):
            try:
                known.append(abs(float(token.rstrip("%"))))
            except ValueError:
                continue
    missing = []
    for token in _numbers(answer):
        value = abs(float(token.rstrip("%")))
        places = _decimals(token)
        if token.rstrip("%") == str(int(value)) and value <= STRUCTURAL_INTEGER_MAX and not token.endswith("%"):
            continue
        tolerance = 0.5 * 10 ** (-places) + 1e-12
        supported = any(abs(k - value) <= tolerance for k in known)
        if not supported and token.endswith("%"):
            supported = any(abs(100 * k - value) <= tolerance for k in known)
        if not supported:
            missing.append(token)
    return missing


# ---------------------------------------------------------------------------
# The tool loop.
# ---------------------------------------------------------------------------

def _default_client():
    import anthropic  # optional dependency: only `ask --llm` needs it

    return anthropic.Anthropic()


def _run_tool(record: RunRecord, name: str, arguments: dict) -> tuple[str, bool]:
    lookup = LOOKUPS.get(name)
    if lookup is None:
        return json.dumps({"error": f"Unknown tool '{name}'."}), True
    try:
        return json.dumps(lookup(record, **(arguments or {})), default=str), False
    except TypeError as exc:  # arguments that do not fit the lookup
        return json.dumps({"error": f"Bad arguments for '{name}': {exc}"}), True


def _fallback(tracking_uri: str, run_id: str, question: str, note: str,
              calls: list[dict] | None = None, unsupported: list[str] | None = None) -> LLMAnswer:
    return LLMAnswer(text=answer_question(tracking_uri, run_id, question), source="keyword",
                     grounded=True, unsupported_numbers=unsupported or [], tool_calls=calls or [], note=note)


def answer_with_llm(tracking_uri: str, run_id: str, question: str, *, client=None,
                    model: str = DEFAULT_MODEL, max_turns: int = MAX_TOOL_TURNS) -> LLMAnswer:
    record = RunRecord(tracking_uri, run_id)
    if not record.exists:
        return _fallback(tracking_uri, run_id, question, "No logged run to ground an answer in.")
    if client is None:
        try:
            client = _default_client()
        except Exception as exc:  # noqa: BLE001 - no SDK or no credentials: the router still works
            return _fallback(tracking_uri, run_id, question,
                             f"LLM unavailable ({type(exc).__name__}: {exc}); answered by the keyword router.")

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    # The question is deliberately NOT evidence: "did it score 0.99?" answered "yes, 0.99"
    # would otherwise verify itself.
    evidence: list[str] = []
    calls: list[dict[str, Any]] = []
    for _ in range(max_turns):
        try:
            response = client.beta.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, tools=TOOLS,
                messages=messages, betas=[FALLBACK_BETA],
                extra_body={"fallbacks": "default", "output_config": {"effort": "medium"}},
            )
        except Exception as exc:  # noqa: BLE001 - any API failure degrades to the router
            return _fallback(tracking_uri, run_id, question,
                             f"LLM request failed ({type(exc).__name__}: {exc}); answered by the keyword router.",
                             calls)

        if response.stop_reason == "refusal":
            return _fallback(tracking_uri, run_id, question,
                             "The model declined this question; answered by the keyword router.", calls)
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                output, is_error = _run_tool(record, block.name, block.input)
                calls.append({"tool": block.name, "input": block.input, "error": is_error})
                evidence.extend([output, json.dumps(block.input)])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output,
                                **({"is_error": True} if is_error else {})})
            messages.append({"role": "user", "content": results})
            continue
        if response.stop_reason != "end_turn":
            return _fallback(tracking_uri, run_id, question,
                             f"The model stopped early ({response.stop_reason}); answered by the keyword router.",
                             calls)

        text = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
        missing = unsupported_numbers(text, evidence)
        if missing:
            return _fallback(
                tracking_uri, run_id, question,
                "The model's answer contained numbers no logged artifact supports ("
                + ", ".join(missing) + "), so it was discarded; answered by the keyword router.",
                calls, missing,
            )
        return LLMAnswer(text=text, source="llm", grounded=True, tool_calls=calls)

    return _fallback(tracking_uri, run_id, question,
                     f"No answer within {max_turns} tool rounds; answered by the keyword router.", calls)
