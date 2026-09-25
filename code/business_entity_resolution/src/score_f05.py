"""
Official-style F_0.5 scorer for the Business Entity Resolution challenge.

Reproduces the challenge's macro-averaged, precision-heavy F_0.5 metric:
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
computed per Source-1 entity, then averaged. Singletons (no true matches)
score 1.0 for an empty prediction and 0.0 for any spurious prediction.

Usage:
    python3 score_f05.py \
        --predictions output/val_matching_results.tsv \
        --ground-truth data/validation/val_ground_truth.tsv \
        --error-report experiments/val_errors.tsv   # optional
"""
from pathlib import Path
import argparse

import pandas as pd


def parse_ids(value: str) -> set[str]:
    value = value.strip()
    if not value:
        return set()
    return set(value.split(","))


def load_map(path: Path, id_col: str, match_col: str) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row[id_col]: parse_ids(row[match_col]) for _, row in df.iterrows()}


def score_entity(predicted: set[str], truth: set[str]) -> tuple[float, float, float]:
    if not truth:
        # Singleton: full credit only for an exactly empty prediction.
        f05 = 1.0 if not predicted else 0.0
        precision = 1.0 if not predicted else 0.0
        recall = 1.0
        return precision, recall, f05

    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth)

    if precision == 0.0 and recall == 0.0:
        return precision, recall, 0.0

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return precision, recall, f05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--error-report", type=Path, default=None)
    args = parser.parse_args()

    truth_map = load_map(args.ground_truth, "source1_entity_id", "matched_entity_ids")
    pred_map = load_map(args.predictions, "source1_entity_id", "matched_entity_ids")

    missing = set(truth_map) - set(pred_map)
    if missing:
        print(
            f"WARNING: {len(missing):,} ground-truth entities missing from predictions "
            f"(scored as empty prediction)."
        )

    rows = []
    for s1_id, truth in truth_map.items():
        predicted = pred_map.get(s1_id, set())
        precision, recall, f05 = score_entity(predicted, truth)
        rows.append(
            {
                "source1_entity_id": s1_id,
                "is_singleton": not truth,
                "n_true": len(truth),
                "n_pred": len(predicted),
                "precision": precision,
                "recall": recall,
                "f0_5": f05,
            }
        )

    results = pd.DataFrame(rows)

    macro_f05 = results["f0_5"].mean()
    macro_precision = results["precision"].mean()
    macro_recall = results["recall"].mean()

    singleton = results[results["is_singleton"]]
    non_singleton = results[~results["is_singleton"]]

    print("=" * 70)
    print("VALIDATION SCORE")
    print("=" * 70)
    print(f"Entities scored:        {len(results):,}")
    print(f"Macro F_0.5:            {macro_f05:.4%}")
    print(f"Macro precision:        {macro_precision:.4%}")
    print(f"Macro recall:           {macro_recall:.4%}")
    print()
    print(
        f"Singletons:             {len(singleton):,} "
        f"(mean F_0.5: {singleton['f0_5'].mean():.4%})"
    )
    print(
        f"Non-singletons:         {len(non_singleton):,} "
        f"(mean F_0.5: {non_singleton['f0_5'].mean():.4%})"
    )

    if args.error_report:
        args.error_report.parent.mkdir(parents=True, exist_ok=True)
        results.sort_values("f0_5").to_csv(args.error_report, sep="\t", index=False)
        print(f"\nPer-entity error report written to: {args.error_report}")


if __name__ == "__main__":
    main()
