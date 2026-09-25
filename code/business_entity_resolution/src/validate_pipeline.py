"""
validate_pipeline.py — one-command wrapper around the team's validation
scripts. Does NOT reimplement any scoring logic itself; it just invokes the
existing tools with consistent defaults, so every blocker/model run gets
checked the same way, and the underlying math only lives in one place each.

Wraps:
    --candidates  -> score_blocking_recall.py   (untouched)
    --predictions -> error_buckets.py           (macro F_0.5 + error buckets)
    --scores      -> threshold_sweep.py

--ground-truth defaults to data/validation/val_ground_truth.tsv and
--val-source1 defaults to data/validation/val_source1.tsv.

Usage:
    python3 validate_pipeline.py \
        --candidates output/candidate_pairs.tsv \
        --predictions output/matching_results.tsv \
        --scores output/candidate_scores.tsv \
        --thresholds 0.10:0.90:0.05
"""
from pathlib import Path
import argparse
import subprocess
import sys

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_GROUND_TRUTH = DATA_DIR / "validation" / "val_ground_truth.tsv"
DEFAULT_VAL_SOURCE1 = DATA_DIR / "validation" / "val_source1.tsv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "experiments" / "pipeline_eval"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--thresholds", default="0.10:0.90:0.05")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--val-source1", type=Path, default=DEFAULT_VAL_SOURCE1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not (args.candidates or args.predictions or args.scores):
        parser.error("Pass at least one of --candidates, --predictions, --scores.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    if args.candidates:
        run([
            python, str(SRC_DIR / "score_blocking_recall.py"),
            "--candidates", str(args.candidates),
            "--ground-truth", str(args.ground_truth),
            "--label", "validate_pipeline run",
        ])

    if args.predictions:
        cmd = [
            python, str(SRC_DIR / "error_buckets.py"),
            "--predictions", str(args.predictions),
            "--ground-truth", str(args.ground_truth),
            "--out", str(args.out_dir / "prediction_errors_categorized.tsv"),
        ]
        if args.val_source1.exists():
            cmd += ["--val-source1", str(args.val_source1)]
        run(cmd)

    if args.scores:
        run([
            python, str(SRC_DIR / "threshold_sweep.py"),
            "--scores", str(args.scores),
            "--ground-truth", str(args.ground_truth),
            "--thresholds", args.thresholds,
            "--out", str(args.out_dir / "threshold_sweep.tsv"),
        ])


if __name__ == "__main__":
    main()
