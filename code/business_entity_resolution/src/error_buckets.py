"""
error_buckets.py — generalized prediction evaluator + error-bucket
categorizer. Accepts ANY predictions file in matching_results.tsv format
(source1_entity_id \t matched_entity_ids), not tied to any one experiment.
Reuses score_f05.py's scoring functions (load_map, score_entity) rather
than reimplementing the F_0.5 math, so the two can't drift apart.

Reports:
    - Macro precision / recall / F_0.5
    - Singleton vs non-singleton performance
    - Error category breakdown: total_miss, under_matched, over_matched,
      mixed_errors, correct, singleton_correct, singleton_false_positive
    - Optional country breakdown (pass --val-source1)

Usage:
    python3 error_buckets.py \
        --predictions output/matching_results.tsv \
        --ground-truth data/validation/val_ground_truth.tsv \
        --val-source1 data/validation/val_source1.tsv \
        --out experiments/errors_categorized.tsv
"""
from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_f05 import load_map, score_entity  # noqa: E402


def categorize(row: pd.Series) -> str:
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


def evaluate(predictions_path: Path, ground_truth_path: Path) -> pd.DataFrame:
    truth_map = load_map(ground_truth_path, "source1_entity_id", "matched_entity_ids")
    pred_map = load_map(predictions_path, "source1_entity_id", "matched_entity_ids")

    rows = []
    for s1_id, truth in truth_map.items():
        predicted = pred_map.get(s1_id, set())
        precision, recall, f05 = score_entity(predicted, truth)
        rows.append({
            "source1_entity_id": s1_id,
            "is_singleton": not truth,
            "n_true": len(truth),
            "n_pred": len(predicted),
            "precision": precision,
            "recall": recall,
            "f0_5": f05,
        })
    results = pd.DataFrame(rows)
    results["category"] = results.apply(categorize, axis=1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--val-source1", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate(args.predictions, args.ground_truth)

    macro_f05 = results["f0_5"].mean()
    macro_precision = results["precision"].mean()
    macro_recall = results["recall"].mean()
    singleton = results[results["is_singleton"]]
    non_singleton = results[~results["is_singleton"]]

    print("=" * 70)
    print("PREDICTION EVALUATION")
    print("=" * 70)
    print(f"Entities scored:        {len(results):,}")
    print(f"Macro F_0.5:            {macro_f05:.4%}")
    print(f"Macro precision:        {macro_precision:.4%}")
    print(f"Macro recall:           {macro_recall:.4%}")
    print(f"Singletons:             {len(singleton):,} (mean F_0.5: {singleton['f0_5'].mean():.4%})")
    print(f"Non-singletons:         {len(non_singleton):,} (mean F_0.5: {non_singleton['f0_5'].mean():.4%})")

    print("\n" + "=" * 70)
    print("ERROR CATEGORY BREAKDOWN")
    print("=" * 70)
    summary = (
        results.groupby("category")
        .agg(count=("source1_entity_id", "size"), mean_f0_5=("f0_5", "mean"))
        .sort_values("count", ascending=False)
    )
    print(summary.to_string())

    if args.val_source1 and args.val_source1.exists():
        s1 = pd.read_csv(args.val_source1, sep="\t", dtype=str, keep_default_na=False)[
            ["entity_id", "business_name", "country"]
        ]
        results = results.merge(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
        print("\n" + "=" * 70)
        print("CATEGORY x COUNTRY")
        print("=" * 70)
        print(pd.crosstab(results["category"], results["country"]).to_string())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.sort_values("f0_5").to_csv(args.out, sep="\t", index=False)
        print(f"\nFull categorized report written to: {args.out}")


if __name__ == "__main__":
    main()
