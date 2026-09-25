from pathlib import Path
from collections import defaultdict, Counter
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = 100_000

MIN_TOKEN_LEN = 4
MAX_TOKEN_FREQUENCY = 50_000


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("&", "and")


def tokenize(name: str) -> set[str]:
    return {
        token
        for token in normalize_name(name).split()
        if len(token) >= MIN_TOKEN_LEN
    }


def count_token_frequency(path: Path) -> Counter:
    frequencies = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for name in chunk["business_name"]:
            frequencies.update(tokenize(name))

    return frequencies


def build_country_token_index(
    path: Path,
    valid_tokens: set[str],
):
    """
    Build:
        country -> token -> entity IDs
    """

    index = defaultdict(lambda: defaultdict(list))

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for entity_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"],
        ):
            tokens = tokenize(name) & valid_tokens

            for token in tokens:
                index[country][token].append(entity_id)

    return index


def load_ground_truth(path: Path):
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
            ground_truth[s1_id] = (
                set(matched.split(","))
                if matched
                else set()
            )

    return ground_truth


def evaluate(
    s1_path,
    index_s2,
    index_s3,
    token_frequency,
    ground_truth,
):
    total_s1 = 0
    total_candidates = 0
    total_true_matches = 0
    recovered_matches = 0
    s1_with_candidates = 0

    start_time = time.time()

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for s1_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"],
        ):
            total_s1 += 1

            tokens = tokenize(name)

            # Only tokens that exist in the frequency index.
            tokens = [
                token
                for token in tokens
                if token in token_frequency
                and token_frequency[token] <= MAX_TOKEN_FREQUENCY
            ]

            # Most selective token first.
            if tokens:
                rarest_token = min(
                    tokens,
                    key=token_frequency.get,
                )
            else:
                rarest_token = None

            candidates = set()

            if rarest_token is not None:
                candidates.update(
                    index_s2.get(country, {}).get(
                        rarest_token,
                        ()
                    )
                )

                candidates.update(
                    index_s3.get(country, {}).get(
                        rarest_token,
                        ()
                    )
                )

            total_candidates += len(candidates)

            if candidates:
                s1_with_candidates += 1

            true_matches = ground_truth[s1_id]

            total_true_matches += len(true_matches)

            recovered_matches += len(
                candidates.intersection(true_matches)
            )

        if total_s1 % 100_000 == 0:
            elapsed = time.time() - start_time

            current_recall = (
                recovered_matches / total_true_matches
                if total_true_matches
                else 0
            )

            print(
                f"Processed {total_s1:,} / 2,206,821 S1 | "
                f"{elapsed / 60:.1f} min | "
                f"Current recall: {current_recall:.4%}"
            )

    recall = (
        recovered_matches / total_true_matches
        if total_true_matches
        else 0
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("EXP-003: COUNTRY-AWARE RAREST-TOKEN BLOCKING")
    print("=" * 70)

    print(f"S1 entities:             {total_s1:,}")
    print(f"S1 with candidates:      {s1_with_candidates:,}")

    print(
        f"Average candidates/S1:   "
        f"{total_candidates / total_s1:.2f}"
    )

    print(f"True matches:             {total_true_matches:,}")
    print(f"Recovered true matches:   {recovered_matches:,}")
    print(f"Blocking recall:          {recall:.4%}")
    print(f"Evaluation time:          {elapsed / 60:.1f} minutes")


def main():
    s2_path = DATA_DIR / "train" / "train_source2.tsv"
    s3_path = DATA_DIR / "train" / "train_source3.tsv"

    print("Counting S2 token frequencies...")
    freq_s2 = count_token_frequency(s2_path)

    print("Counting S3 token frequencies...")
    freq_s3 = count_token_frequency(s3_path)

    combined_frequency = freq_s2 + freq_s3

    valid_tokens = {
        token
        for token, frequency in combined_frequency.items()
        if frequency <= MAX_TOKEN_FREQUENCY
    }

    print(f"\nUnique tokens: {len(combined_frequency):,}")
    print(f"Usable tokens: {len(valid_tokens):,}")

    print("\nBuilding country-aware S2 index...")
    index_s2 = build_country_token_index(
        s2_path,
        valid_tokens,
    )

    print("Building country-aware S3 index...")
    index_s3 = build_country_token_index(
        s3_path,
        valid_tokens,
    )

    print("\nLoading ground truth...")
    ground_truth = load_ground_truth(
        DATA_DIR / "train" / "train_ground_truth.tsv"
    )

    print("\nEvaluating...")

    evaluate(
        DATA_DIR / "train" / "train_source1.tsv",
        index_s2,
        index_s3,
        combined_frequency,
        ground_truth,
    )


if __name__ == "__main__":
    main()