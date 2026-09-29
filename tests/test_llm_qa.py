"""
T2-5 — an LLM front-end whose answers are checked before they are shown.

No network here: a scripted client stands in for the API and plays the model's
side of the tool loop. What is under test is everything this project owns — the
tools dispatch to the grounded lookups, the answer is verified against what the
tools returned, and every failure degrades to the keyword router instead of to a
wrong answer or an exception.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoeng.explain.llm_qa import TOOLS, answer_with_llm, unsupported_numbers
from autoeng.explain.qa import LOOKUPS, RunRecord, lookup_candidate
from autoeng.tracking.mlflow_tracker import log_pipeline_run


def _payload():
    return {
        "source_path": "data.csv",
        "ingestion_report": {"file_format": "csv", "detected_encoding": "utf-8", "warnings": []},
        "profile_summary": {"n_rows": 500, "n_cols": 4, "columns": {}},
        "problem_decision": {"chosen": {"problem_type": "binary_classification", "target_column": "churned",
                                        "time_column": None, "reasoning": ["two classes"], "score": 0.9},
                             "confidence": 0.88,
                             "alternatives": [{"problem_type": "regression", "target_column": "income",
                                               "score": 0.31}]},
        "structural_cleaning_report": {"actions": ["Dropped 3 duplicate rows."]},
        "role_assignment": {},
        "pre_training_leakage": {"flags": [{"severity": "warning", "kind": "proxy", "columns": ["refund"],
                                            "description": "refund is recorded after churn"}]},
        "leaderboard": {"problem_kind": "classification", "primary_metric": "roc_auc", "results": [
            {"name": "logistic_regression", "status": "ok", "metrics": {"roc_auc": 0.8123},
             "fit_time_seconds": 1.0, "error": None, "evaluation_stage": "full"},
            {"name": "random_forest", "status": "ok", "metrics": {"roc_auc": 0.7911},
             "fit_time_seconds": 3.0, "error": None, "evaluation_stage": "full"},
            {"name": "qda", "status": "failed", "metrics": {}, "fit_time_seconds": 0.0,
             "error": "LinAlgError: singular matrix", "evaluation_stage": "full"},
        ]},
        "hpo_results": [],
        "post_training_leakage": {"flags": []},
        "explanation": {"winner_name": "logistic_regression", "winner_score": 0.8123,
                        "runner_up_name": "random_forest", "runner_up_score": 0.7911, "margin": 0.0212,
                        "cv_fold_std": 0.03, "margin_within_noise": True, "hpo_improvement": None,
                        "feature_importances": [["age", 0.41], ["income", 0.22], ["city", 0.05]],
                        "importance_method": "shap", "narrative": "logistic_regression won narrowly."},
        "final_report_text": "# test",
    }


@pytest.fixture
def run(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_id = log_pipeline_run(tracking_uri=uri, run_name="qa", **_payload())
    return uri, run_id


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _call(name, arguments, call_id="t1"):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


class _Scripted:
    """Plays the model: returns the scripted responses in order, recording each request."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        stop_reason, content = self.responses.pop(0)
        return SimpleNamespace(stop_reason=stop_reason, content=content)


class TestGroundingCheck:
    EVIDENCE = ['{"roc_auc": 0.8123, "gap_to_winner": 0.0212, "n_rows": 1961}']

    def test_numbers_the_tools_returned_pass(self):
        assert unsupported_numbers("It scored 0.8123, a gap of 0.0212.", self.EVIDENCE) == []

    def test_rounding_to_fewer_places_passes(self):
        assert unsupported_numbers("Roughly 0.81, a gap of 0.02.", self.EVIDENCE) == []

    def test_a_percentage_of_a_returned_number_passes(self):
        assert unsupported_numbers("About 81% ROC-AUC.", self.EVIDENCE) == []

    def test_thousands_separators_are_read_as_one_number(self):
        assert unsupported_numbers("Trained on 1,961 rows.", self.EVIDENCE) == []

    def test_an_invented_number_is_caught(self):
        assert unsupported_numbers("It scored 0.8123 against 0.8500.", self.EVIDENCE) == ["0.8500"]

    def test_false_precision_is_caught(self):
        # 0.8129 is not what the tool said, even though 0.81 would have been fine.
        assert unsupported_numbers("It scored 0.8129.", self.EVIDENCE) == ["0.8129"]

    def test_small_counts_are_structural_and_names_are_not_numbers(self):
        assert unsupported_numbers("The top 3 features by F1 of xgboost2.", self.EVIDENCE) == []


class TestTools:
    def test_every_tool_is_backed_by_a_lookup(self):
        assert {t["name"] for t in TOOLS} == set(LOOKUPS)
        for tool in TOOLS:
            schema = tool["input_schema"]
            assert tool["strict"] and schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])

    def test_a_candidate_lookup_carries_its_gap_to_the_winner(self, run):
        found = lookup_candidate(RunRecord(*run), "random forest")
        assert found["name"] == "random_forest" and found["gap_to_winner"] == pytest.approx(0.0212)


class TestTheLoop:
    def test_a_grounded_answer_is_returned_with_its_tool_calls(self, run):
        client = _Scripted(
            ("tool_use", [_call("candidate", {"model": "random_forest"})]),
            ("end_turn", [_text("random_forest scored 0.7911, 0.0212 behind logistic_regression (0.8123).")]),
        )
        answer = answer_with_llm(*run, "How far behind was the forest?", client=client)
        assert answer.source == "llm" and answer.grounded
        assert answer.tool_calls == [{"tool": "candidate", "input": {"model": "random_forest"}, "error": False}]
        request = client.requests[0]
        assert request["model"] == "claude-opus-5"
        assert request["extra_body"]["fallbacks"] == "default"
        # The tool result went back to the model tied to its call.
        result = client.requests[1]["messages"][-1]["content"][0]
        assert result["tool_use_id"] == "t1" and "0.7911" in result["content"]

    def test_an_answer_with_an_invented_number_is_discarded(self, run):
        client = _Scripted(
            ("tool_use", [_call("winner", {})]),
            ("end_turn", [_text("logistic_regression won with 0.8123, well clear of the 0.7500 runner-up.")]),
        )
        answer = answer_with_llm(*run, "Why did logistic regression win?", client=client)
        assert answer.source == "keyword" and answer.unsupported_numbers == ["0.7500"]
        assert "0.7500" in answer.note and "discarded" in answer.note
        assert "0.7500" not in answer.text

    def test_a_number_taken_from_the_question_does_not_verify_itself(self, run):
        client = _Scripted(("end_turn", [_text("Yes, random_forest scored 0.99.")]))
        answer = answer_with_llm(*run, "Did random_forest score 0.99?", client=client)
        assert answer.source == "keyword" and answer.unsupported_numbers == ["0.99"]

    def test_an_honest_i_dont_know_passes(self, run):
        client = _Scripted(("end_turn", [_text("This run did not log anything about training time.")]))
        answer = answer_with_llm(*run, "How long did training take?", client=client)
        assert answer.source == "llm"

    def test_an_unknown_tool_is_an_error_result_not_a_crash(self, run):
        client = _Scripted(
            ("tool_use", [_call("delete_everything", {})]),
            ("end_turn", [_text("I could not find that.")]),
        )
        answer = answer_with_llm(*run, "anything", client=client)
        result = client.requests[1]["messages"][-1]["content"][0]
        assert result["is_error"] is True and answer.tool_calls[0]["error"] is True

    def test_a_refusal_falls_back_to_the_router(self, run):
        answer = answer_with_llm(*run, "Is there leakage?", client=_Scripted(("refusal", [])))
        assert answer.source == "keyword" and "declined" in answer.note
        assert "refund is recorded after churn" in answer.text

    def test_a_tool_loop_that_never_ends_falls_back(self, run):
        loop = [("tool_use", [_call("leaderboard", {"top_n": 10}, f"t{i}")]) for i in range(3)]
        answer = answer_with_llm(*run, "Which model won?", client=_Scripted(*loop), max_turns=3)
        assert answer.source == "keyword" and "3 tool rounds" in answer.note

    def test_an_api_error_falls_back(self, run):
        class _Broken:
            beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(
                ConnectionError("network down"))))

        answer = answer_with_llm(*run, "Which model won?", client=_Broken())
        assert answer.source == "keyword" and "network down" in answer.note

    def test_no_sdk_or_credentials_falls_back(self, run, monkeypatch):
        from autoeng.explain import llm_qa

        def _no_client():
            raise RuntimeError("no credentials")

        monkeypatch.setattr(llm_qa, "_default_client", _no_client)
        answer = answer_with_llm(*run, "Which features mattered?")
        assert answer.source == "keyword" and "no credentials" in answer.note
        assert "age" in answer.text

    def test_a_missing_run_says_so_without_calling_the_model(self, tmp_path):
        uri = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
        client = _Scripted()
        answer = answer_with_llm(uri, "nope", "anything", client=client)
        assert answer.source == "keyword" and client.requests == []
