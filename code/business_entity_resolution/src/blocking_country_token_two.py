from pathlib import Path
from collections import Counter, defaultdict
import time

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = 100_000

MIN_TOKEN_LEN = 4
MAX_TOKEN_FREQUENCY = 50_000

# EXP-004:
# Use the 2 rarest usable tokens from each Source-1 name.
MAX_TOKENS_PER_RECORD = 2


# ============================================================
# Normalization
# ============================================================

def normalize_name(name: str) -> str:
    return name.lower().strip().replace("&", "and")


def tokenize(name: str) -> set[str]:
    return {
        token
        for token in normalize_name(name).split()
        if len(token) >= MIN_TOKEN_LEN
    }


# ============================================================
# Pass 1:
# Count token frequencies across Source 2 + Source 3
# ============================================================

def count_token_frequencies() -> Counter:
    print("=" * 70)
    print("PASS 1: COUNTING TOKEN FREQUENCIES")
    print("=" * 70)

    token_frequency = Counter()

    source_paths = [
        DATA_DIR / "train" / "train_source2.tsv",
        DATA_DIR / "train" / "train_source3.tsv",
    ]

    for source_path in source_paths:
        print(f"\nReading: {source_path}")

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):
            for name in chunk["business_name"]:
                tokens = tokenize(name)

                # Count each token once per record.
                token_frequency.update(tokens)

    print(f"\nUnique tokens: {len(token_frequency):,}")

    usable_tokens = {
        token
        for token, frequency in token_frequency.items()
        if frequency <= MAX_TOKEN_FREQUENCY
    }

    print(f"Usable tokens: {len(usable_tokens):,}")

    return token_frequency


# ============================================================
# Pass 2:
# Build country -> token -> entity IDs indexes
# ============================================================

def build_indexes(token_frequency: Counter):
    print("\n" + "=" * 70)
    print("PASS 2: BUILDING COUNTRY-AWARE TOKEN INDEXES")
    print("=" * 70)

    index_s2 = defaultdict(lambda: defaultdict(set))
    index_s3 = defaultdict(lambda: defaultdict(set))

    usable_tokens = {
        token
        for token, frequency in token_frequency.items()
        if frequency <= MAX_TOKEN_FREQUENCY
    }

    source_configs = [
        (
            DATA_DIR / "train" / "train_source2.tsv",
            index_s2,
            "Source 2",
        ),
        (
            DATA_DIR / "train" / "train_source3.tsv",
            index_s3,
            "Source 3",
        ),
    ]

    for source_path, index, source_name in source_configs:
        print(f"\nBuilding index for {source_name}: {source_path}")

        processed = 0

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):
            for row in chunk.itertuples(index=False):
                entity_id = row.entity_id
                country = row.country
                tokens = tokenize(row.business_name)

                for token in tokens:
                    if token in usable_tokens:
                        index[country][token].add(entity_id)

            processed += len(chunk)

            if processed % 1_000_000 < CHUNK_SIZE:
                print(f"  Processed: {processed:,}")

    print("\nIndexes built.")

    return index_s2, index_s3


# ============================================================
# Load ground truth
# ============================================================

def load_ground_truth():
    print("\n" + "=" * 70)
    print("LOADING GROUND TRUTH")
    print("=" * 70)

    ground_truth_path = (
        DATA_DIR / "train" / "train_ground_truth.tsv"
    )

    ground_truth = {}

    for chunk in pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            value = row.matched_entity_ids.strip()

            if value:
                matches = set(value.split(","))
            else:
                matches = set()

            ground_truth[row.source1_entity_id] = matches

    print(f"Ground-truth S1 entities: {len(ground_truth):,}")

    return ground_truth


# ============================================================
# EXP-004 Evaluation
# ============================================================

def evaluate(index_s2, index_s3, ground_truth):
    print("\n" + "=" * 70)
    print("EXP-004: COUNTRY + TWO RAREST TOKENS")
    print("=" * 70)

    source1_path = DATA_DIR / "train" / "train_source1.tsv"

    total_s1 = 0
    s1_with_candidates = 0

    total_candidates = 0

    total_true_matches = 0
    recovered_true_matches = 0

    start_time = time.time()

    for chunk in pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):

            total_s1 += 1

            entity_id = row.entity_id
            country = row.country

            tokens = tokenize(row.business_name)

            # Keep only tokens that exist in the frequency index.
            usable_tokens = [
                token
                for token in tokens
                if token in token_frequency_global
                and token_frequency_global[token] <= MAX_TOKEN_FREQUENCY
            ]

            # ------------------------------------------------
            # Select TWO rarest tokens
            # ------------------------------------------------

            rare_tokens = sorted(
                usable_tokens,
                key=token_frequency_global.get,
            )[:MAX_TOKENS_PER_RECORD]

            candidates = set()

            # ------------------------------------------------
            # Country-aware candidate retrieval
            #
            # IMPORTANT:
            # This is UNION / OR logic.
            # A candidate matching either rare token is included.
            # ------------------------------------------------

            for token in rare_tokens:

                candidates.update(
                    index_s2
                    .get(country, {})
                    .get(token, ())
                )

                candidates.update(
                    index_s3
                    .get(country, {})
                    .get(token, ())
                )

            candidate_count = len(candidates)

            total_candidates += candidate_count

            if candidate_count > 0:
                s1_with_candidates += 1

            # ------------------------------------------------
            # Blocking recall
            # ------------------------------------------------

            true_matches = ground_truth.get(
                entity_id,
                set(),
            )

            total_true_matches += len(true_matches)

            recovered_true_matches += len(
                true_matches.intersection(candidates)
            )

        if total_s1 % 500_000 < CHUNK_SIZE:
            elapsed = time.time() - start_time

            print(
                f"S1 processed: {total_s1:,} | "
                f"Avg candidates/S1: "
                f"{total_candidates / total_s1:,.2f} | "
                f"Elapsed: {elapsed / 60:.1f} min"
            )

    elapsed = time.time() - start_time

    average_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0
    )

    blocking_recall = (
        recovered_true_matches / total_true_matches
        if total_true_matches
        else 0
    )

    candidate_coverage = (
        s1_with_candidates / total_s1
        if total_s1
        else 0
    )

    # ========================================================
    # Final results
    # ========================================================

    print("\n" + "=" * 70)
    print("EXP-004 RESULTS")
    print("=" * 70)

    print(f"Training S1 entities:       {total_s1:,}")
    print(f"S1 with candidates:         {s1_with_candidates:,}")
    print(f"S1 candidate coverage:      {candidate_coverage:.4%}")
    print(f"Total candidates:           {total_candidates:,}")
    print(f"Average candidates / S1:    {average_candidates:,.2f}")
    print(f"True matches:               {total_true_matches:,}")
    print(f"Recovered true matches:     {recovered_true_matches:,}")
    print(f"Blocking recall:            {blocking_recall:.4%}")
    print(f"Evaluation time:            {elapsed / 60:.1f} min")

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    overall_start = time.time()

    # Pass 1
    token_frequency_global = count_token_frequencies()

    # Pass 2
    index_s2, index_s3 = build_indexes(
        token_frequency_global
    )

    # Ground truth
    ground_truth = load_ground_truth()

    # Evaluation
    evaluate(
        index_s2,
        index_s3,
        ground_truth,
    )

    total_time = time.time() - overall_start

    print(
        f"\nTotal experiment runtime: "
        f"{total_time / 60:.1f} minutes"
    )