"""
Vocabulary coverage: the one text measure training and monitoring must agree on.

T2-3 made models read the words of a text column, but drift had no way to see
the words change — a new error message, new slang, a renamed product leave
length and word count where they were. Coverage is the share of a document's
words that the training corpus already knew, where "knew" means the word
appeared in at least `MIN_DOCUMENT_FREQUENCY` training documents (a word seen
once is noise, not vocabulary).

The reference has to be honest about how often ordinary new documents contain
unknown words, or every live window looks drifted. So each TRAINING document is
scored as if it were new: a word counts as known only if it appears in at least
`MIN_DOCUMENT_FREQUENCY` OTHER training documents (leave one out). A live
document is scored against the vocabulary of all of them, which is the same rule.

Tokens follow scikit-learn's default token pattern, lower-cased, so this is the
same notion of a word the TF-IDF features use.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

_TOKEN = re.compile(r"(?u)\b\w\w+\b")
MIN_DOCUMENT_FREQUENCY = 2
#: Same ceiling as the TF-IDF vocabulary (`TEXT_MAX_VOCABULARY`).
MAX_VOCABULARY = 20_000


def tokenize(text: object) -> list[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return _TOKEN.findall(str(text).lower())


def build_vocabulary(texts: Iterable[object]) -> tuple[list[str], np.ndarray]:
    """(vocabulary, leave-one-out coverage per document).

    Coverage is NaN for a document with no tokens: it has no words to be known
    or unknown, and counting it as 0 or 1 would invent a value.
    """
    documents = [tokenize(t) for t in texts]
    frequency = Counter(term for doc in documents for term in set(doc))
    known = [t for t, n in frequency.items() if n >= MIN_DOCUMENT_FREQUENCY]
    known.sort(key=lambda t: (-frequency[t], t))
    vocabulary = set(known[:MAX_VOCABULARY])
    coverage = np.full(len(documents), np.nan)
    for i, doc in enumerate(documents):
        if doc:
            present = set(doc)
            # Known without this document: in the vocabulary, and in enough OTHER documents.
            ok = {t for t in present if t in vocabulary and frequency[t] - 1 >= MIN_DOCUMENT_FREQUENCY}
            coverage[i] = sum(t in ok for t in doc) / len(doc)
    return sorted(vocabulary), coverage


def coverage(texts: Iterable[object], vocabulary: Iterable[str]) -> np.ndarray:
    """Share of each document's tokens that are in `vocabulary` (NaN when it has none)."""
    known = set(vocabulary)
    values = []
    for text in texts:
        doc = tokenize(text)
        values.append(sum(t in known for t in doc) / len(doc) if doc else np.nan)
    return np.asarray(values, dtype=float)
