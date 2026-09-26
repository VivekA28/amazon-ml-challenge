from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
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
    ap.add_argument("--negative-ratio", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=500_000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    usecols = ["source1_entity_id", "label"] + FEATURES

    sampled_parts = []
    total_rows = 0
    total_pos = 0
    total_neg = 0

    print("Streaming feature file...")

    for chunk in pd.read_csv(
        args.features,
        sep="\t",
        usecols=usecols,
        chunksize=args.chunk_size,
    ):
        total_rows += len(chunk)

        pos = chunk[chunk["label"] == 1]
        neg = chunk[chunk["label"] == 0]

        total_pos += len(pos)
        total_neg += len(neg)

        # Keep every positive.
        sampled_parts.append(pos)

        # Sample negatives approximately at the requested ratio.
        if len(neg) > 0:
            n_take = min(
                len(neg),
                max(1, len(pos) * args.negative_ratio),
            )

            if n_take < len(neg):
                idx = rng.choice(
                    len(neg),
                    size=n_take,
                    replace=False,
                )
                sampled_parts.append(
                    neg.iloc[idx]
                )
            else:
                sampled_parts.append(neg)

        if total_rows % (args.chunk_size * 10) < args.chunk_size:
            print(
                f"Processed: {total_rows:,} rows | "
                f"sampled so far: "
                f"{sum(len(x) for x in sampled_parts):,}"
            )

    print()
    print(f"All feature rows: {total_rows:,}")
    print(f"Original positives: {total_pos:,}")
    print(f"Original negatives: {total_neg:,}")

    train = pd.concat(sampled_parts, ignore_index=True)

    # Free the individual chunk references.
    sampled_parts.clear()

    # Compact numeric representation.
    X = train[FEATURES].astype(np.float32)
    y = train["label"].astype(np.int8)

    print(f"Training rows: {len(train):,}")
    print(f"Training positives: {int(y.sum()):,}")
    print(f"Training negatives: {int((y == 0).sum()):,}")

    del train

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=4,
        random_state=args.seed,
    )

    print("Starting LightGBM training...")

    model.fit(X, y)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model))

    print(f"Saved model: {args.model}")


if __name__ == "__main__":
    main()