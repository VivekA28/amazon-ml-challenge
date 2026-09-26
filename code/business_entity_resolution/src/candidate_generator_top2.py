from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import re
import time

import pandas as pd


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


def name_tokens(value: str) -> list[str]:
    return [x for x in value.split() if len(x) >= MIN_TOKEN_LEN]


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


def count_name_tokens(paths):
    counts = Counter()

    for path in paths:
        print(f"[freq] {path.name}", flush=True)
        started = time.time()

        for chunk in read_source(path):
            for name in chunk["business_name"].map(norm):
                # One count per record/token.
                counts.update(set(name_tokens(name)))

        print(
            f"[freq] done in {(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    return counts


def build_indexes(paths, token_freq):
    exact_name = defaultdict(set)
    country_name = defaultdict(set)
    country_number = defaultdict(set)
    country_addr = defaultdict(set)

    usable_name_tokens = {
        tok
        for tok, freq in token_freq.items()
        if len(tok) >= MIN_TOKEN_LEN and freq <= MAX_TOKEN_FREQ
    }

    for path in paths:
        print(f"[index] {path.name}", flush=True)
        started = time.time()

        for chunk in read_source(path):
            names = chunk["business_name"].map(norm)
            addresses = chunk["business_address"].map(norm)

            for eid, country_raw, name, address in zip(
                chunk["entity_id"],
                chunk["country"],
                names,
                addresses,
            ):
                country = norm(country_raw)

                # ---------------------------------------------------------
                # Exact normalized name index
                # ---------------------------------------------------------
                if name:
                    exact_name[name].add(eid)

                    # -----------------------------------------------------
                    # Country + usable name-token index
                    # -----------------------------------------------------
                    for tok in name_tokens(name):
                        if tok in usable_name_tokens:
                            country_name[(country, tok)].add(eid)

                # ---------------------------------------------------------
                # Address indexes
                # ---------------------------------------------------------
                if address:
                    number = addr_number(address)

                    if number:
                        country_number[(country, number)].add(eid)

                    for tok in name_tokens(address):
                        key = (country, tok)
                        posting = country_addr.get(key)

                        # Do not retain huge address-token postings.
                        if posting is None:
                            country_addr[key] = {eid}
                        elif len(posting) <= MAX_TOKEN_FREQ:
                            posting.add(eid)

        print(
            f"[index] done in {(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    return (
        exact_name,
        country_name,
        country_number,
        country_addr,
    )


def top_usable_name_tokens(
    name: str,
    token_freq: Counter[str],
) -> list[str]:
    """
    Return the two rarest usable name tokens.

    A token is usable when:
      - length >= MIN_TOKEN_LEN
      - global document frequency <= MAX_TOKEN_FREQ

    Ties are broken lexicographically, matching the V1 logic.
    """
    usable = [
        tok
        for tok in name_tokens(name)
        if token_freq.get(tok, MAX_TOKEN_FREQ + 1) <= MAX_TOKEN_FREQ
    ]

    usable.sort(key=lambda tok: (token_freq[tok], tok))

    return usable[:2]


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
        out.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

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

                # =========================================================
                # Layer A:
                # Exact normalized-name candidates
                # =========================================================
                if name:
                    candidates.update(
                        exact_name.get(name, ())
                    )

                # =========================================================
                # Layers B + C:
                # Top-2 rare usable name-token anchors
                # =========================================================
                top_tokens = top_usable_name_tokens(
                    name,
                    token_freq,
                )

                for name_tok in top_tokens:

                    name_posting = country_name.get(
                        (country, name_tok),
                        (),
                    )

                    if not name_posting:
                        continue

                    # -----------------------------------------------------
                    # Layer B:
                    # Same country
                    # + same address number
                    # + same name token
                    # -----------------------------------------------------
                    number = addr_number(address)

                    if number:
                        number_posting = country_number.get(
                            (country, number),
                            (),
                        )

                        if number_posting:

                            # Iterate over the smaller posting list.
                            if len(name_posting) < len(number_posting):
                                candidates.update(
                                    x
                                    for x in name_posting
                                    if x in number_posting
                                )
                            else:
                                candidates.update(
                                    x
                                    for x in number_posting
                                    if x in name_posting
                                )

                    # -----------------------------------------------------
                    # Layer C:
                    # Same country
                    # + shared address token
                    # + same name token
                    #
                    # Choose the smallest address-token posting before
                    # intersecting, matching V1's optimization.
                    # -----------------------------------------------------
                    if address:

                        best_addr_posting = None

                        for addr_tok in name_tokens(address):
                            posting = country_addr.get(
                                (country, addr_tok)
                            )

                            if posting and (
                                best_addr_posting is None
                                or len(posting)
                                < len(best_addr_posting)
                            ):
                                best_addr_posting = posting

                        if best_addr_posting:

                            if len(name_posting) < len(
                                best_addr_posting
                            ):
                                candidates.update(
                                    x
                                    for x in name_posting
                                    if x in best_addr_posting
                                )
                            else:
                                candidates.update(
                                    x
                                    for x in best_addr_posting
                                    if x in name_posting
                                )

                # =========================================================
                # Write deduplicated candidates
                # =========================================================
                ids = sorted(candidates)

                if ids:
                    with_candidates += 1
                    total_candidates += len(ids)

                out.write(
                    f"{eid}\t{','.join(ids)}\n"
                )

            print(
                f"[gen] processed {total_s1:,} S1 | "
                f"pairs {total_candidates:,}",
                flush=True,
            )

    print("\n" + "=" * 78)
    print("candidate_generator_top2 COMPLETE")
    print("=" * 78)
    print(f"S1 entities          : {total_s1:,}")
    print(f"S1 with candidates   : {with_candidates:,}")
    print(
        f"Candidate coverage   : "
        f"{100 * with_candidates / total_s1:.4f}%"
    )
    print(f"Total candidate pairs: {total_candidates:,}")
    print(
        f"Average / S1         : "
        f"{total_candidates / total_s1:.2f}"
    )
    print(f"Output               : {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1",
        type=Path,
        required=True,
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
        required=True,
    )

    args = parser.parse_args()

    started = time.time()

    paths = [
        args.source2,
        args.source3,
    ]

    print("=" * 78)
    print("candidate_generator_top2")
    print("=" * 78)

    # =============================================================
    # 1. Count global name-token frequencies
    # =============================================================
    print("\n[1/3] Counting name-token frequencies...")

    token_freq = count_name_tokens(paths)

    print(
        f"[freq] unique tokens: {len(token_freq):,}",
        flush=True,
    )

    # =============================================================
    # 2. Build candidate indexes
    # =============================================================
    print("\n[2/3] Building indexes...")

    indexes = build_indexes(
        paths,
        token_freq,
    )

    # =============================================================
    # 3. Generate validation candidates
    # =============================================================
    print("\n[3/3] Generating candidates...")

    generate_candidates(
        args.s1,
        *indexes,
        token_freq,
        args.output,
    )

    print(
        f"\nTotal runtime: "
        f"{(time.time() - started) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()