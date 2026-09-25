from pathlib import Path
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = 100_000


def normalize_name(name: str) -> str:
    """Basic normalization for the first blocking experiment."""
    return (
        name.lower()
        .strip()
        .replace("&", "and")
    )


def build_name_index(path: Path) -> dict[str, list[str]]:
    """Build normalized-name -> entity IDs index for S2/S3."""
    index = defaultdict(list)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        names = chunk["business_name"].map(normalize_name)

        for name, entity_id in zip(names, chunk["entity_id"]):
            index[name].append(entity_id)

    return index


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    """Load S1 -> true matched IDs."""
    ground_truth = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for s1_id, matched in zip(
            chunk["source1_entity_id"],
            chunk["matched_entity_ids"],
        ):
            if matched:
                ground_truth[s1_id] = set(matched.split(","))
            else:
                ground_truth[s1_id] = set()

    return ground_truth


def evaluate_source1(
    path: Path,
    index_s2: dict[str, list[str]],
    index_s3: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
) -> None:

    total = 0
    total_candidates = 0
    total_true_matches = 0
    recovered_matches = 0

    s1_with_candidates = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for s1_id, name in zip(
            chunk["entity_id"],
            chunk["business_name"],
        ):
            total += 1

            normalized = normalize_name(name)

            candidates = (
                index_s2.get(normalized, [])
                + index_s3.get(normalized, [])
            )

            total_candidates += len(candidates)

            if candidates:
                s1_with_candidates += 1

            true_matches = ground_truth[s1_id]

            total_true_matches += len(true_matches)
            recovered_matches += len(
                set(candidates) & true_matches
            )

    print("\n" + "=" * 70)
    print("EXACT NORMALIZED-NAME BLOCKING")
    print("=" * 70)

    print(f"S1 entities:             {total:,}")
    print(f"S1 with candidates:      {s1_with_candidates:,}")
    print(
        f"Average candidates/S1:  "
        f"{total_candidates / total:.2f}"
    )

    print(
        f"True matches:            "
        f"{total_true_matches:,}"
    )

    print(
        f"Recovered true matches:  "
        f"{recovered_matches:,}"
    )

    recall = (
        recovered_matches / total_true_matches
        if total_true_matches
        else 0
    )

    print(f"Blocking recall:         {recall:.4%}")


def main():
    print("Building S2 name index...")
    index_s2 = build_name_index(
        DATA_DIR / "train" / "train_source2.tsv"
    )

    print("Building S3 name index...")
    index_s3 = build_name_index(
        DATA_DIR / "train" / "train_source3.tsv"
    )

    print("Loading ground truth...")
    ground_truth = load_ground_truth(
        DATA_DIR / "train" / "train_ground_truth.tsv"
    )

    print("Evaluating S1...")
    evaluate_source1(
        DATA_DIR / "train" / "train_source1.tsv",
        index_s2,
        index_s3,
        ground_truth,
    )


if __name__ == "__main__":
    main()