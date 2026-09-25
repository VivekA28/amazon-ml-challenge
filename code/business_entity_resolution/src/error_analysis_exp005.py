"""
Categorize EXP-005 val_errors.tsv into meaningful failure buckets, broken
down by country, to guide feature engineering (Priority 4) and later
blocking work.

Buckets:
    singleton_correct        - true singleton, correctly predicted empty
    singleton_false_positive - true singleton, but we predicted a match
    nonsingleton_total_miss  - had true matches, predicted nothing
    nonsingleton_correct     - precision == 1.0 and recall == 1.0
    nonsingleton_over_matched  - recall == 1.0, precision < 1.0 (extra wrong candidates)
    nonsingleton_under_matched - precision == 1.0, recall < 1.0 (missed some true matches)
    nonsingleton_mixed_errors  - both wrong candidates AND missed matches

Usage:
    python3 error_analysis_exp005.py \
        --errors ../../../experiments/val_errors.tsv \
        --val-source1 ../../../data/validation/val_source1.tsv \
        --out ../../../experiments/val_errors_categorized.tsv
"""
from pathlib import Path
import argparse

import pandas as pd


def categorize(row) -> str:
    if row["is_singleton"]:
        return "singleton_correct" if row["n_pred"] == 0 else "singleton_false_positive"
    if row["n_pred"] == 0:
        return "nonsingleton_total_miss"
    if row["precision"] == 1.0 and row["recall"] == 1.0:
        return "nonsingleton_correct"
    if row["recall"] == 1.0 and row["precision"] < 1.0:
        return "nonsingleton_over_matched"
    if row["precision"] == 1.0 and row["recall"] < 1.0:
        return "nonsingleton_under_matched"
    return "nonsingleton_mixed_errors"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", required=True, type=Path)
    parser.add_argument("--val-source1", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    errors = pd.read_csv(args.errors, sep="\t", dtype=str, keep_default_na=False)
    for col in ("n_true", "n_pred"):
        errors[col] = errors[col].astype(int)
    for col in ("precision", "recall", "f0_5"):
        errors[col] = errors[col].astype(float)
    errors["is_singleton"] = errors["is_singleton"].map({"True": True, "False": False})

    s1 = pd.read_csv(
        args.val_source1, sep="\t", dtype=str, keep_default_na=False
    )[["entity_id", "business_name", "country"]]

    merged = errors.merge(
        s1, left_on="source1_entity_id", right_on="entity_id", how="left"
    )
    merged["category"] = merged.apply(categorize, axis=1)

    print("=" * 70)
    print("ERROR CATEGORY BREAKDOWN")
    print("=" * 70)
    summary = (
        merged.groupby("category")
        .agg(count=("source1_entity_id", "size"), mean_f0_5=("f0_5", "mean"))
        .sort_values("count", ascending=False)
    )
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("CATEGORY x COUNTRY")
    print("=" * 70)
    cross = pd.crosstab(merged["category"], merged["country"])
    print(cross.to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values("f0_5").to_csv(args.out, sep="\t", index=False)
    print(f"\nFull categorized report written to: {args.out}")


if __name__ == "__main__":
    main()
