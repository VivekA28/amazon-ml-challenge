"""
Entity-safe validation split for the Business Entity Resolution challenge.

Splits Source-1 entities (and their ground-truth rows) into TRAIN and VAL.
Source-2 / Source-3 pools are NOT split — blocking/matching must still search
the full S2/S3 universe on both splits, exactly as at test time. This is safe
because each S2/S3 record matches at most one Source-1 entity (see the
ground-truth match-count profiling already done), so splitting on S1 alone
cannot leak S2/S3 records between splits.

Usage:
    python3 create_validation_split.py --val-fraction 0.15 --seed 42
"""
from pathlib import Path
import argparse
import random

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "validation"

CHUNK_SIZE = 100_000


def load_ground_truth(path: Path) -> pd.DataFrame:
    chunks = pd.read_csv(
        path, sep="\t", dtype=str, chunksize=CHUNK_SIZE, keep_default_na=False
    )
    return pd.concat(chunks, ignore_index=True)


def split_ids(ids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    return train_ids, val_ids


def write_filtered_source1(path: Path, ids: set[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, chunksize=CHUNK_SIZE, keep_default_na=False
    ):
        subset = chunk[chunk["entity_id"].isin(ids)]
        subset.to_csv(
            out_path, sep="\t", index=False, mode="w" if first else "a", header=first
        )
        first = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading training ground truth...")
    gt = load_ground_truth(DATA_DIR / "train" / "train_ground_truth.tsv")

    all_ids = gt["source1_entity_id"].tolist()
    print(f"Total S1 entities: {len(all_ids):,}")

    train_ids, val_ids = split_ids(all_ids, args.val_fraction, args.seed)
    print(f"Train split: {len(train_ids):,} entities")
    print(f"Val split:   {len(val_ids):,} entities")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Writing split ground truth files...")
    gt_train = gt[gt["source1_entity_id"].isin(train_ids)]
    gt_val = gt[gt["source1_entity_id"].isin(val_ids)]
    gt_train.to_csv(OUT_DIR / "train_ground_truth.tsv", sep="\t", index=False)
    gt_val.to_csv(OUT_DIR / "val_ground_truth.tsv", sep="\t", index=False)

    print("Writing split Source-1 files...")
    write_filtered_source1(
        DATA_DIR / "train" / "train_source1.tsv", train_ids, OUT_DIR / "train_source1.tsv"
    )
    write_filtered_source1(
        DATA_DIR / "train" / "train_source1.tsv", val_ids, OUT_DIR / "val_source1.tsv"
    )

    n_singleton_val = (gt_val["matched_entity_ids"].str.strip() == "").sum()
    print(f"\nVal singleton rate: {n_singleton_val / len(gt_val):.2%}")
    print("Source2/Source3 files are NOT split — for blocking/matching on either")
    print("split, keep using the full train_source2.tsv / train_source3.tsv pools.")
    print(f"\nOutput written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
