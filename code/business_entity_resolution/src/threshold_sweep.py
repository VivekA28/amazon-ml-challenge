"""
threshold_sweep.py — sweep classification thresholds over per-candidate-pair
model scores and report macro F_0.5/precision/recall at each one. Built
ahead of the model existing, so a model just needs to produce scores in
this format to plug straight in.

Input format for --scores (tab-separated):
    source1_entity_id \t candidate_entity_id \t score

Usage:
    python3 threshold_sweep.py \
        --scores output/candidate_scores.tsv \
        --ground-truth data/validation/val_ground_truth.tsv \
        --thresholds 0.10:0.90:0.05 \
        --out experiments/threshold_sweep.tsv
"""
from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_f05 import load_map, score_entity  # noqa: E402

CHUNK_SIZE = 100_000


def load_pair_scores(path: Path) -> dict[str, list[tuple[str, float]]]:
    scores: dict[str, list[tuple[str, float]]] = {}
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, chunksize=CHUNK_SIZE, keep_default_na=False
    ):
        for s1_id, cand_id, score in zip(
            chunk["source1_entity_id"], chunk["candidate_entity_id"], chunk["score"]
        ):
            scores.setdefault(s1_id, []).append((cand_id, float(score)))
    return scores


def parse_thresholds(spec: str) -> list[float]:
    start, stop, step = (float(x) for x in spec.split(":"))
    thresholds = []
    t = start
    while t <= stop + 1e-9:
        thresholds.append(round(t, 4))
        t += step
    return thresholds


def macro_metrics_at_threshold(
    pair_scores: dict[str, list[tuple[str, float]]],
    ground_truth: dict[str, set[str]],
    threshold: float,
) -> tuple[float, float, float]:
    f05_list, precision_list, recall_list = [], [], []
    for s1_id, truth in ground_truth.items():
        pairs = pair_scores.get(s1_id, [])
        predicted = {cid for cid, score in pairs if score >= threshold}
        precision, recall, f05 = score_entity(predicted, truth)
        f05_list.append(f05)
        precision_list.append(precision)
        recall_list.append(recall)
    n = len(f05_list) or 1
    return sum(f05_list) / n, sum(precision_list) / n, sum(recall_list) / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--thresholds", default="0.10:0.90:0.05")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ground_truth = load_map(args.ground_truth, "source1_entity_id", "matched_entity_ids")
    pair_scores = load_pair_scores(args.scores)
    thresholds = parse_thresholds(args.thresholds)

    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    rows = []
    for threshold in thresholds:
        macro_f05, macro_precision, macro_recall = macro_metrics_at_threshold(
            pair_scores, ground_truth, threshold
        )
        rows.append({
            "threshold": threshold,
            "macro_f0_5": macro_f05,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
        })
        print(f"  threshold={threshold:.2f}  F_0.5={macro_f05:.4%}  "
              f"precision={macro_precision:.4%}  recall={macro_recall:.4%}")

    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["macro_f0_5"].idxmax()]
    print(f"\nBest threshold: {best['threshold']} (macro F_0.5 = {best['macro_f0_5']:.4%})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(args.out, sep="\t", index=False)
        print(f"Sweep table written to: {args.out}")


if __name__ == "__main__":
    main()
