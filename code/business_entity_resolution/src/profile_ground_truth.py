from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GROUND_TRUTH = PROJECT_ROOT / "data" / "train" / "train_ground_truth.tsv"

CHUNK_SIZE = 100_000


def main():
    match_counts = Counter()
    total = 0

    for chunk in pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for value in chunk["matched_entity_ids"]:
            if not value.strip():
                count = 0
            else:
                count = len(value.split(","))

            match_counts[count] += 1
            total += 1

    print("=" * 70)
    print("GROUND TRUTH MATCH DISTRIBUTION")
    print("=" * 70)

    print(f"Total S1 entities: {total:,}\n")

    print("Match count distribution:")

    for count in sorted(match_counts):
        number = match_counts[count]
        percentage = number / total * 100

        print(
            f"  {count:>3} matches : "
            f"{number:>10,} "
            f"({percentage:6.2f}%)"
        )


if __name__ == "__main__":
    main()