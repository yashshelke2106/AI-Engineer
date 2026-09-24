"""
Generates data/synthetic_text.csv — a support-ticket dataset whose signal lives
in the *words*, for exercising T2-3 (TF-IDF -> SVD features) offline.

The measurements behind T2-3 were taken on real 20-newsgroups posts, which are
not committed here (they need a download, and this repo's fixtures are meant to
work offline). This generator gives the same shape: free text that decides the
label, alongside a categorical and a numeric column that only partly explain it,
so a run has to use the text to do well.

Deliberately NOT length-matched: real tickets differ in length, and the point of
the fixture is to look like real data. `tests/test_text_features.py` uses a
length-matched corpus instead, because a test that proves the words are used
must leave the length features no signal at all.

    python scripts/generate_text.py [--rows 900] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_text.csv"

# Escalating tickets talk about money and repetition; routine ones ask questions.
ANGRY_OPENERS = [
    "this is the third time i am writing about",
    "i have been charged twice for",
    "nobody has replied to my messages about",
    "i want a refund for",
    "your team promised a fix last week for",
]
CALM_OPENERS = [
    "quick question about",
    "could you please explain",
    "i am trying to understand",
    "wondering if it is possible to",
    "just checking on",
]
SUBJECTS = [
    "the invoice on my account", "the subscription renewal", "the data export feature",
    "the mobile app login", "the monthly usage report", "the api rate limit",
    "the seat i removed last month", "the discount that was applied",
]
ANGRY_TAILS = [
    "and i still have no answer", "this is unacceptable", "i am considering cancelling",
    "please escalate this to a manager", "i expect this resolved today",
]
CALM_TAILS = [
    "thanks in advance", "no rush at all", "whenever you get a chance",
    "happy to send screenshots", "thanks for the help",
]
CHANNELS = ["email", "chat", "phone", "portal"]


def build(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(rows):
        # The latent state is how angry the ticket is; the words express it, the
        # numeric and categorical columns only hint at it.
        angry = rng.random() < 0.42
        openers, tails = (ANGRY_OPENERS, ANGRY_TAILS) if angry else (CALM_OPENERS, CALM_TAILS)
        text = " ".join([
            str(rng.choice(openers)), str(rng.choice(SUBJECTS)), str(rng.choice(tails)),
        ])
        # A few tickets mix registers, so the words are informative but not a giveaway.
        if rng.random() < 0.18:
            text += " " + str(rng.choice(CALM_TAILS if angry else ANGRY_TAILS))
        response_hours = float(np.round(rng.gamma(2.0, 6.0 if angry else 3.0), 2))
        channel = str(rng.choice(CHANNELS, p=[0.4, 0.3, 0.1, 0.2]))
        # Escalation follows the anger, the wait, and phone contact — with noise,
        # so nothing here is separable without reading the ticket.
        logit = (1.6 * angry) + 0.05 * response_hours + (0.5 if channel == "phone" else 0.0) - 2.1
        escalated = int(rng.random() < 1 / (1 + np.exp(-logit)))
        records.append({
            "ticket_text": text,
            "channel": channel,
            "response_hours": response_hours,
            "escalated": escalated,
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    frame = build(args.rows, args.seed)
    frame.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {args.out} — {len(frame)} rows, "
          f"{frame['escalated'].mean():.1%} escalated, "
          f"{frame['ticket_text'].nunique()} distinct tickets")


if __name__ == "__main__":
    main()
