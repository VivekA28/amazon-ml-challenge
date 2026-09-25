from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import re
import time

import pandas as pd


# Project layout expected:
# <project_root>/
#   data/train/train_source1.tsv
#   data/train/train_source2.tsv
#   data/train/train_source3.tsv
#   data/validation/validation_source1.tsv   (optional; otherwise --s1 is used)
#   output/candidate_pairs.tsv
#
# This script intentionally keeps the candidate set as the final set that will
# later be scored by the matching model.

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = 100_000
MIN_TOKEN_LEN = 4
MAX_TOKEN_FREQ = 50_000

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\b\d+[a-z]?\b")


def norm(value: object) -> str:
    value = "" if value is None else str(value)
    value = value.lower().strip().replace("&", "and")
    value = _NON_ALNUM.sub(" ", value)
    return " ".join(value.split())


def name_tokens(value: str) -> set[str]:
    return {tok for tok in value.split() if len(tok) >= MIN_TOKEN_LEN}


def addr_number(value: str) -> str | None:
    m = _NUMBER.search(value)
    return m.group(0) if m else None


def read_source(path: Path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    )


def count_name_tokens(source_paths: list[Path]) -> Counter[str]:
    """Exact global token frequencies over S2+S3."""
    counts: Counter[str] = Counter()

    for path in source_paths:
        started = time.time()
        print(f"[freq] {path.name}", flush=True)

        for chunk in read_source(path):
            names = chunk["business_name"].map(norm)
            for name in names:
                # Count a token once per record, matching document frequency.
                counts.update(name_tokens(name))

        print(f"[freq] done in {(time.time() - started) / 60:.1f} min", flush=True)

    return counts


def build_indexes(source_paths: list[Path], token_freq: Counter[str]):
    """
    Build only the indexes needed by candidate_generator_v1.

    exact_name:
        normalized_name -> entity IDs

    country_name:
        (country, rare_name_token) -> entity IDs

    country_number:
        (country, address_number) -> entity IDs

    country_addr:
        (country, address_token) -> entity IDs

    Address-token postings are capped at MAX_TOKEN_FREQ-equivalent volume
    while building to avoid pathological high-frequency postings.
    """
    exact_name: dict[str, set[str]] = defaultdict(set)
    country_name: dict[tuple[str, str], set[str]] = defaultdict(set)
    country_number: dict[tuple[str, str], set[str]] = defaultdict(set)
    country_addr: dict[tuple[str, str], set[str]] = defaultdict(set)

    usable_name_tokens = {
        tok for tok, freq in token_freq.items()
        if len(tok) >= MIN_TOKEN_LEN and freq <= MAX_TOKEN_FREQ
    }

    for path in source_paths:
        started = time.time()
        print(f"[index] {path.name}", flush=True)

        for chunk in read_source(path):
            names = chunk["business_name"].map(norm)
            addrs = chunk["business_address"].map(norm)

            for eid, country_raw, name, address in zip(
                chunk["entity_id"],
                chunk["country"],
                names,
                addrs,
            ):
                country = norm(country_raw)

                if name:
                    exact_name[name].add(eid)

                    for tok in name_tokens(name):
                        if tok in usable_name_tokens:
                            country_name[(country, tok)].add(eid)

                if address:
                    number = addr_number(address)
                    if number:
                        country_number[(country, number)].add(eid)

                    for tok in name_tokens(address):
                        key = (country, tok)
                        posting = country_addr.get(key)

                        # Do not retain huge address-token postings. Such tokens
                        # are not useful for this blocker and would consume a
                        # large amount of RAM.
                        if posting is None:
                            country_addr[key] = {eid}
                        elif len(posting) <= MAX_TOKEN_FREQ:
                            posting.add(eid)

        print(f"[index] done in {(time.time() - started) / 60:.1f} min", flush=True)

    return exact_name, country_name, country_number, country_addr


def rarest_usable_name_token(name: str, token_freq: Counter[str]) -> str | None:
    usable = [
        tok for tok in name_tokens(name)
        if token_freq.get(tok, MAX_TOKEN_FREQ + 1) <= MAX_TOKEN_FREQ
    ]
    if not usable:
        return None
    return min(usable, key=lambda tok: (token_freq[tok], tok))


def generate_candidates(
    s1_path: Path,
    exact_name,
    country_name,
    country_number,
    country_addr,
    token_freq: Counter[str],
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_s1 = 0
    with_candidates = 0
    total_candidates = 0

    with output_path.open("w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk in read_source(s1_path):
            for eid, country_raw, raw_name, raw_address in zip(
                chunk["entity_id"],
                chunk["country"],
                chunk["business_name"],
                chunk["business_address"],
            ):
                total_s1 += 1

                country = norm(country_raw)
                name = norm(raw_name)
                address = norm(raw_address)

                candidates: set[str] = set()

                # Layer A: exact normalized name.
                if name:
                    candidates.update(exact_name.get(name, ()))

                # Use the rarest usable name token as the anchor for the
                # country-aware address/name blocks.
                rare_tok = rarest_usable_name_token(name, token_freq)

                if rare_tok is not None:
                    name_posting = country_name.get((country, rare_tok), ())

                    # Layer B: same country + address number + rare name token.
                    number = addr_number(address)
                    if number and name_posting:
                        number_posting = country_number.get((country, number), ())
                        if number_posting:
                            if len(name_posting) < len(number_posting):
                                candidates.update(
                                    x for x in name_posting if x in number_posting
                                )
                            else:
                                candidates.update(
                                    x for x in number_posting if x in name_posting
                                )

                    # Layer C: same country + address token + rare name token.
                    # Pick the smallest usable address-token posting.
                    if address and name_posting:
                        best_addr_posting = None
                        for atok in name_tokens(address):
                            posting = country_addr.get((country, atok))
                            if posting and (
                                best_addr_posting is None
                                or len(posting) < len(best_addr_posting)
                            ):
                                best_addr_posting = posting

                        if best_addr_posting:
                            if len(name_posting) < len(best_addr_posting):
                                candidates.update(
                                    x for x in name_posting if x in best_addr_posting
                                )
                            else:
                                candidates.update(
                                    x for x in best_addr_posting if x in name_posting
                                )

                ids = sorted(candidates)
                if ids:
                    with_candidates += 1
                    total_candidates += len(ids)

                out.write(f"{eid}\t{','.join(ids)}\n")

    avg = total_candidates / total_s1 if total_s1 else 0.0
    coverage = 100.0 * with_candidates / total_s1 if total_s1 else 0.0

    print("\n" + "=" * 78)
    print("candidate_generator_v1 COMPLETE")
    print("=" * 78)
    print(f"S1 entities          : {total_s1:,}")
    print(f"S1 with candidates   : {with_candidates:,}")
    print(f"Candidate coverage   : {coverage:.4f}%")
    print(f"Total candidate pairs: {total_candidates:,}")
    print(f"Average / S1         : {avg:.2f}")
    print(f"Output               : {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s1",
        type=Path,
        default=DATA_DIR / "train" / "validation_source1.tsv",
        help="S1 TSV to generate candidates for.",
    )
    parser.add_argument(
        "--source2",
        type=Path,
        default=DATA_DIR / "train" / "train_source2.tsv",
    )
    parser.add_argument(
        "--source3",
        type=Path,
        default=DATA_DIR / "train" / "train_source3.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "candidate_pairs_validation.tsv",
    )
    args = parser.parse_args()

    source_paths = [args.source2, args.source3]

    overall = time.time()
    print("=" * 78)
    print("candidate_generator_v1")
    print("=" * 78)
    print(f"S1     : {args.s1}")
    print(f"S2     : {args.source2}")
    print(f"S3     : {args.source3}")
    print(f"Output : {args.output}")

    print("\n[1/3] Counting exact global name-token document frequencies...")
    token_freq = count_name_tokens(source_paths)

    print("\n[2/3] Building candidate-generation indexes...")
    indexes = build_indexes(source_paths, token_freq)

    print("\n[3/3] Generating final candidate set...")
    generate_candidates(
        args.s1,
        *indexes,
        token_freq,
        args.output,
    )

    print(f"\nTotal runtime: {(time.time() - overall) / 60:.1f} min")


if __name__ == "__main__":
    main()
