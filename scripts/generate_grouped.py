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
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.customers
    trait = rng.normal(0, 1, n)
    fingerprint = rng.normal(0, 1, n)
    home_region = rng.choice(["north", "south", "east", "west", "central"], n)
    converted = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-1.0 * trait))).astype(int)

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
