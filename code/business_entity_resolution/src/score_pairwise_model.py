#!/usr/bin/env python3
"""
Score candidate feature rows with the trained LightGBM model.

Output format matches threshold_sweep.py:
    source1_entity_id
    candidate_entity_id
    score
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

FEATURES = [
    "name_exact",
    "name_core_exact",
    "name_jaccard",
    "name_overlap",
    "name_ratio",
    "address_exact",
    "address_jaccard",
    "address_overlap",
    "address_ratio",
    "address_number_equal",
    "country_equal",
    "name_address_mean",
    "name_address_min",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--chunksize", type=int, default=250_000)
    args = ap.parse_args()

    model = lgb.Booster(model_file=str(args.model))
    first = True

    for chunk in pd.read_csv(
        args.features,
        sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str},
        chunksize=args.chunksize,
    ):
        scores = model.predict(chunk[FEATURES])

        out = pd.DataFrame(
            {
                "source1_entity_id": chunk["source1_entity_id"],
                "candidate_entity_id": chunk["candidate_entity_id"],
                "score": scores,
            }
        )

        out.to_csv(
            args.output,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        first = False

        print(f"Scored {len(chunk):,} rows", flush=True)

    print(f"Saved scores: {args.output}")


if __name__ == "__main__":
    main()
