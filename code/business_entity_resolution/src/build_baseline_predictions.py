"""
Trivial rule-based baseline: every EXP-001 exact-normalized-name candidate
(from either S2 or S3) is emitted directly as a predicted match, no further
filtering. This is NOT a blocking experiment (see EXP-001/EXP-002 for that,
owned separately) -- it exists purely to push a real, imperfect prediction
file through score_f05.py, so we get a first genuine F_0.5 number and can
trust the scorer before a real model exists.

Runs against the VAL split only (data/validation/val_source1.tsv), scored
against the FULL train_source2/3 pools, matching how validation should work.

Usage:
    python3 build_baseline_predictions.py
"""
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

CHUNK_SIZE = 100_000


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("&", "and")


def build_name_index(path: Path) -> dict[str, list[str]]:
    index = defaultdict(list)
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, chunksize=CHUNK_SIZE, keep_default_na=False
    ):
        for name, entity_id in zip(chunk["business_name"].map(normalize_name), chunk["entity_id"]):
            index[name].append(entity_id)
    return index


def main():
    print("Building S2 name index (full pool)...")
    index_s2 = build_name_index(DATA_DIR / "train" / "train_source2.tsv")

    print("Building S3 name index (full pool)...")
    index_s3 = build_name_index(DATA_DIR / "train" / "train_source3.tsv")

    print("Scoring val split S1 entities...")
    val_s1 = pd.read_csv(
        DATA_DIR / "validation" / "val_source1.tsv", sep="\t", dtype=str, keep_default_na=False
    )

    rows = []
    for s1_id, name in zip(val_s1["entity_id"], val_s1["business_name"]):
        normalized = normalize_name(name)
        candidates = index_s2.get(normalized, []) + index_s3.get(normalized, [])
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(candidates)})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "val_matching_results.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(rows):,} predictions to {out_path}")


if __name__ == "__main__":
    main()
