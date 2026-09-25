from pathlib import Path
import argparse
import pandas as pd

CHUNK_SIZE = 100_000


def load_ground_truth(path):
    truth = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            truth[row.source1_entity_id] = (
                set(row.matched_entity_ids.split(","))
                if row.matched_entity_ids
                else set()
            )

    return truth


def evaluate(candidates_path, ground_truth_path):
    print("Loading ground truth...")
    truth = load_ground_truth(ground_truth_path)

    total_true_matches = sum(len(v) for v in truth.values())

    recovered = 0
    total_s1 = 0
    s1_with_candidates = 0
    total_candidates = 0

    for chunk in pd.read_csv(
        candidates_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            total_s1 += 1

            candidate_ids = (
                set(row.candidate_entity_ids.split(","))
                if row.candidate_entity_ids
                else set()
            )

            if candidate_ids:
                s1_with_candidates += 1

            total_candidates += len(candidate_ids)

            true_ids = truth.get(row.source1_entity_id, set())
            recovered += len(true_ids & candidate_ids)

    recall = (
        recovered / total_true_matches
        if total_true_matches
        else 0.0
    )

    coverage = (
        s1_with_candidates / total_s1
        if total_s1
        else 0.0
    )

    avg_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0.0
    )

    print("\n" + "=" * 78)
    print("BLOCKING RECALL")
    print("=" * 78)

    print(f"S1 entities           : {total_s1:,}")
    print(f"S1 with candidates    : {s1_with_candidates:,}")
    print(f"Candidate coverage    : {coverage:.4%}")
    print(f"Total candidates      : {total_candidates:,}")
    print(f"Average candidates/S1 : {avg_candidates:.2f}")
    print(f"True matches           : {total_true_matches:,}")
    print(f"Recovered matches      : {recovered:,}")
    print(f"Blocking recall        : {recall:.4%}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--label",
        default="blocking",
    )

    args = parser.parse_args()

    print("=" * 78)
    print(args.label)
    print("=" * 78)

    evaluate(
        args.candidates,
        args.ground_truth,
    )


if __name__ == "__main__":
    main()