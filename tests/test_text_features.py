"""
T2-3 — free text as real features, fitted per fold.

Measured before building, on 20-newsgroups posts through this project's own
pipeline: with only length/word-count stats a 6-topic model reached ROC-AUC
0.591 (random forest 0.557) and a 2-topic one 0.674 — the length difference
between topics, and nothing else. Adding TF-IDF -> TruncatedSVD took the same
pipelines to 0.985 and 0.995. Shuffling the text against the label left the
score at chance (0.485 -> 0.490), so the block adds no signal where there is
none. Components plateau at 50: 10 -> 0.977, 50 -> 0.985, 100 -> 0.986.

The corpus here is synthetic on purpose: both classes draw from one shared
vocabulary and differ only in four topic words, and every note is exactly the
same length, so length and word count carry *no* signal by construction and
anything the model finds has to come from the words.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from autoeng.common.roles import assign_feature_roles
from autoeng.features.pipeline_builder import build_preprocessing_pipeline
from autoeng.features.transformers import TextVectorFeaturizer
from autoeng.profiling.profiler import profile_dataset

DATA = Path(__file__).resolve().parents[1] / "data"

# Every word is five letters, so char_len, word_count and avg_word_len are
# IDENTICAL for every note. An earlier draft used natural words and the stats
# features reached ROC-AUC 0.684 on their character lengths alone — the
# vocabulary has to be length-matched for this fixture to prove anything.
SHARED = ["these", "thing", "again", "where", "since", "while", "other", "found", "given", "after"]
TOPIC_A = ["pitch", "bases", "glove", "swing"]
TOPIC_B = ["fever", "nurse", "cough", "tumor"]
WORDS_PER_NOTE = 16


def _corpus(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        topic = TOPIC_A if i % 2 == 0 else TOPIC_B
        words = list(rng.choice(SHARED, WORDS_PER_NOTE - 4)) + list(rng.choice(topic, 4))
        rng.shuffle(words)
        rows.append({"note": " ".join(words), "label": "a" if i % 2 == 0 else "b"})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _pipeline(frame, *, vectorize_text=True):
    profile = profile_dataset(frame)
    roles = assign_feature_roles(profile, target_column="label")
    pipe = build_preprocessing_pipeline(roles, "classification", use_interactions=False,
                                        vectorize_text=vectorize_text)
    return pipe, roles, frame[roles.feature_columns], frame["label"]


def _cv_auc(frame, *, vectorize_text=True):
    pipe, _, X, y = _pipeline(frame, vectorize_text=vectorize_text)
    est = Pipeline(list(pipe.steps) + [("model", LogisticRegression(max_iter=2000))])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    return cross_val_score(est, X, y, cv=cv, scoring="roc_auc").mean()


class TestWordsBecomeFeatures:
    def test_free_text_is_routed_to_text_extraction(self):
        _, roles, _, _ = _pipeline(_corpus(120))
        assert roles.text_columns == ["note"]
        assert "note" not in roles.numeric_columns + roles.categorical_columns

    def test_length_stats_alone_cannot_see_the_signal_but_the_words_can(self):
        frame = _corpus()
        stats_only = _cv_auc(frame, vectorize_text=False)
        with_words = _cv_auc(frame, vectorize_text=True)
        assert stats_only < 0.60, "length-matched notes must leave the stats features blind"
        assert with_words > 0.90
        assert with_words - stats_only > 0.30

    def test_the_raw_column_is_dropped_and_everything_out_is_numeric(self):
        frame = _corpus(120)
        pipe, _, X, y = _pipeline(frame)
        out = pipe.fit_transform(X, y)
        assert "note" not in out.columns
        assert "note__svd_0" in out.columns and "note__word_count" in out.columns
        assert not any(str(d) == "object" for d in out.dtypes), "a string column would break every estimator"
        assert np.isfinite(out.to_numpy(dtype=float)).all()

    def test_switching_vectorisation_off_leaves_the_stats_behind(self):
        pipe, _, X, y = _pipeline(_corpus(120), vectorize_text=False)
        out = pipe.fit_transform(X, y)
        assert not [c for c in out.columns if "svd" in c]
        assert "note__word_count" in out.columns


class TestFittedPerFoldOnly:
    def test_a_word_only_in_held_out_rows_never_enters_the_vocabulary(self):
        frame = _corpus(200)
        train, held_out = frame.iloc[:150].copy(), frame.iloc[150:].copy()
        held_out["note"] = held_out["note"] + " zebrafish"
        featurizer = TextVectorFeaturizer(text_columns=["note"]).fit(train[["note"]])
        vocabulary = featurizer.vectorizers_["note"][0].vocabulary_
        assert "zebrafish" not in vocabulary
        out = featurizer.transform(held_out[["note"]])
        assert np.isfinite(out.filter(like="__svd_").to_numpy()).all(), "unseen words must be ignored, not fatal"

    def test_the_component_count_is_identical_across_transforms(self):
        frame = _corpus(200)
        featurizer = TextVectorFeaturizer(text_columns=["note"]).fit(frame[["note"]])
        wide = featurizer.transform(frame[["note"]]).filter(like="__svd_").shape[1]
        one_row = featurizer.transform(frame[["note"]].head(1)).filter(like="__svd_").shape[1]
        assert wide == one_row, "serving scores one row at a time and must get the same feature space"


class TestDegenerateText:
    def test_clone_round_trips_with_no_text_columns(self):
        # Invariant 2: parameters are stored verbatim, so clone() can rebuild them.
        featurizer = TextVectorFeaturizer()
        assert clone(featurizer).get_params() == featurizer.get_params()
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        assert TextVectorFeaturizer().fit(frame).transform(frame).equals(frame)

    def test_a_vocabulary_too_thin_to_decompose_is_reported_not_crashed(self):
        # Two terms leave one component at most, which is not a decomposition.
        frame = pd.DataFrame({"note": ["alpha beta"] * 20 + ["alpha beta"] * 20})
        featurizer = TextVectorFeaturizer(text_columns=["note"]).fit(frame)
        assert "note" in featurizer.skipped_ and featurizer.skipped_["note"]
        out = featurizer.transform(frame)
        assert not [c for c in out.columns if "__svd_" in c]
        assert "note" in out.columns, "the stats step downstream still needs the raw column"

    def test_text_made_only_of_stop_words_keeps_them_instead_of_raising(self):
        # Pruning English stop words empties this vocabulary, which TfidfVectorizer
        # raises on; the fallback keeps them, so these notes are still usable.
        frame = pd.DataFrame({"note": [f"the and of it {w}" for w in ["one", "two", "six", "ten"] * 10]})
        featurizer = TextVectorFeaturizer(text_columns=["note"]).fit(frame)
        assert "note" not in featurizer.skipped_
        assert featurizer.transform(frame).filter(like="__svd_").shape[1] >= 2

    def test_one_document_repeated_is_not_a_corpus(self):
        frame = pd.DataFrame({"note": ["same words every single time"] * 40})
        featurizer = TextVectorFeaturizer(text_columns=["note"]).fit(frame)
        assert "distinct" in featurizer.skipped_["note"]
        featurizer.transform(frame)  # must not raise

    def test_empty_and_missing_values_are_text_too(self):
        frame = _corpus(120)
        frame.loc[:5, "note"] = ""
        frame.loc[6:10, "note"] = None
        pipe, _, X, y = _pipeline(frame)
        out = pipe.fit_transform(X, y)
        assert np.isfinite(out.to_numpy(dtype=float)).all()

    def test_short_notes_still_get_components_when_the_vocabulary_allows(self):
        # 30 rows, 14 words: fewer rows than the default 50 components.
        frame = _corpus(30)
        featurizer = TextVectorFeaturizer(text_columns=["note"]).fit(frame[["note"]])
        out = featurizer.transform(frame[["note"]])
        made = out.filter(like="__svd_").shape[1]
        assert 2 <= made < 30, f"components must be clamped below the row and vocabulary count, got {made}"


class TestRepetitiveTextIsStillText:
    """Prose below the 95%-unique bar used to become a high-cardinality category.

    Measured on data/synthetic_text.csv (900 tickets, 499 distinct): target
    encoding scored ROC-AUC 0.646 / 0.564 / 0.578 (logistic regression / random
    forest / hist gradient boosting) against 0.650 / 0.532 / 0.586 with the column
    dropped entirely, and 0.695 / 0.636 / 0.642 routed to TF-IDF -> SVD.
    """

    def _profile(self, frame, column):
        return profile_dataset(frame).columns[column].semantic_type

    def test_repeated_multi_word_tickets_are_text_not_categories(self):
        rng = np.random.default_rng(0)
        openers = ["i have been charged twice for", "quick question about", "please escalate"]
        subjects = ["the invoice on my account", "the monthly usage report", "the api rate limit"]
        tails = ["this is unacceptable", "thanks in advance", "no rush at all", "i expect a reply today"]
        frame = pd.DataFrame({"ticket": [
            f"{rng.choice(openers)} {rng.choice(subjects)} {rng.choice(tails)}" for _ in range(600)
        ]})
        assert frame["ticket"].nunique() / len(frame) < 0.95, "this fixture must fail the near-unique bar"
        assert self._profile(frame, "ticket").value == "text_free"

    def test_a_handful_of_long_survey_answers_is_a_category(self):
        answers = ["strongly agree with the statement as written",
                   "somewhat agree with the statement as written",
                   "neither agree nor disagree with the statement",
                   "somewhat disagree with the statement as written"]
        frame = pd.DataFrame({"answer": (answers * 150)})
        assert self._profile(frame, "answer").value == "categorical_low_card"

    def test_many_short_labels_stay_a_high_cardinality_category(self):
        rng = np.random.default_rng(1)
        cities = [f"city_{i}" for i in range(120)]
        frame = pd.DataFrame({"city": rng.choice(cities, 900)})
        assert self._profile(frame, "city").value == "categorical_high_card"

    def test_the_committed_ticket_fixture_reaches_the_text_path(self):
        frame = pd.read_csv(DATA / "synthetic_text.csv")
        roles = assign_feature_roles(profile_dataset(frame), target_column="escalated")
        assert roles.text_columns == ["ticket_text"]
        assert "ticket_text" not in roles.high_card_categorical_columns

    def test_the_words_beat_target_encoding_on_that_fixture(self):
        # The measurement that motivated the routing change, at 3 folds and one
        # model to keep it affordable (~10s).
        frame = pd.read_csv(DATA / "synthetic_text.csv")
        profile = profile_dataset(frame)
        as_text = assign_feature_roles(profile, target_column="escalated")
        as_category = replace(
            as_text,
            text_columns=[],
            categorical_columns=as_text.categorical_columns + ["ticket_text"],
            high_card_categorical_columns=as_text.high_card_categorical_columns + ["ticket_text"],
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        scores = {}
        for label, roles in (("text", as_text), ("category", as_category)):
            pipe = build_preprocessing_pipeline(roles, "classification", use_interactions=False)
            est = Pipeline(list(pipe.steps) + [("model", LogisticRegression(max_iter=2000))])
            scores[label] = cross_val_score(est, frame[roles.feature_columns], frame["escalated"],
                                            cv=cv, scoring="roc_auc").mean()
        assert scores["text"] > scores["category"] + 0.02, scores


def test_the_roles_reasoning_says_the_words_are_used():
    # A reader of the report must not be told the column is only measured for length.
    _, roles, _, _ = _pipeline(_corpus(120))
    reasoning = roles.reasoning["note"].lower()
    assert "tf-idf" in reasoning and "length" in reasoning
