"""
Generate new customers from the process behind data/synthetic_grouped.csv.

The champion-challenger gate scores the challenger on a forward window of
traffic neither model trained on. Replaying the training file again cannot
provide one: every row repeats a vector the challenger was fitted to, and the
gate rightly excludes it. New customers from the same generator can.

    python scripts/generate_grouped.py --customers 60 --seed 11 --out data/new_customers.csv

The generator matches tests/conftest.py's grouped fixture: a customer-level
trait decides conversion, device_fingerprint and home_region identify the
customer, measure_b and measure_c vary by visit.

--concept S (0 to 1) changes the relationship itself, which --shift in
replay_traffic.py cannot: conversion moves from the trait towards
device_fingerprint, which caused nothing in training. Inputs look the same;
what they mean for the outcome does not. That is a concept change, and it is
what a retrained challenger can genuinely win on:

    python scripts/generate_grouped.py --customers 300 --seed 21 --concept 0.7 --out data/concept_a.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--customers", type=int, default=60)
    parser.add_argument("--visits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11, help="Different seeds give different customers.")
    parser.add_argument("--prefix", default="N", help="Customer id prefix, so ids never collide with C0000..C0149.")
    parser.add_argument("--concept", type=float, default=0.0,
                        help="0 = training's relationship; 1 = conversion decided by device_fingerprint instead.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.concept <= 1.0:
        parser.error("--concept must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    n = args.customers
    trait = rng.normal(0, 1, n)
    fingerprint = rng.normal(0, 1, n)
    home_region = rng.choice(["north", "south", "east", "west", "central"], n)
    # Strength matched to the trait's: a unit-variance driver on the same logit
    # scale, so --concept moves the outcome without making it easier or harder.
    logit = (1.0 - args.concept) * trait + args.concept * fingerprint
    converted = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-1.0 * logit))).astype(int)

    columns = ["customer_id", "home_region", "device_fingerprint", "measure_a", "measure_b", "measure_c", "converted"]
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for c in range(n):
            for _ in range(args.visits):
                writer.writerow([
                    f"{args.prefix}{args.seed}-{c:04d}", home_region[c],
                    round(fingerprint[c] + rng.normal(0, 0.01), 4),
                    round(trait[c] + rng.normal(0, 0.01), 4),
                    round(trait[c] * 0.4 + rng.normal(0, 0.9), 4),
                    round(rng.normal(0, 1.0), 4),
                    int(converted[c]),
                ])
    print(f"Wrote {n} customers x {args.visits} visits = {n * args.visits} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
