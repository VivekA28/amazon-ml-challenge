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

# Critical safety cap: never dump a large country+token posting list.
# This keeps candidate generation bounded and fast.
MAX_RARE_TOKEN_POSTING = 100

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\b\d+[a-z]?\b")


def norm(value: object) -> str:
    value = "" if value is None else str(value)
    value = value.lower().strip().replace("&", "and")
    value = _NON_ALNUM.sub(" ", value)
    return " ".join(value.split())


def tokens(value: str) -> list[str]:
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
        start = time.time()

        for chunk in read_source(path):
            for name in chunk["business_name"].map(norm):
                # One count per record/token.
                counts.update(set(tokens(name)))

        print(f"[freq] done in {(time.time()-start)/60:.1f} min", flush=True)

    return counts


def build_indexes(paths, freq):
    exact_name = defaultdict(list)
    country_name = defaultdict(list)
    country_number = defaultdict(list)
    country_addr = defaultdict(list)

    usable = {
        tok for tok, n in freq.items()
        if len(tok) >= MIN_TOKEN_LEN and n <= MAX_TOKEN_FREQ
    }

    for path in paths:
        print(f"[index] {path.name}", flush=True)
        start = time.time()

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

                if name:
                    exact_name[name].append(eid)

                    for tok in set(tokens(name)):
                        if tok in usable:
                            key = (country, tok)
                            # Only retain small postings.
                            if len(country_name[key]) < MAX_RARE_TOKEN_POSTING:
                                country_name[key].append(eid)

                if address:
                    number = addr_number(address)
                    if number:
                        key = (country, number)
                        if len(country_number[key]) < 1000:
                            country_number[key].append(eid)

                    for tok in set(tokens(address)):
                        key = (country, tok)
                        if len(country_addr[key]) < 500:
                            country_addr[key].append(eid)

        print(f"[index] done in {(time.time()-start)/60:.1f} min", flush=True)

    return exact_name, country_name, country_number, country_addr


def rare_tokens(name, freq):
    usable = [
        tok for tok in set(tokens(name))
        if freq.get(tok, MAX_TOKEN_FREQ + 1) <= MAX_TOKEN_FREQ
    ]
    usable.sort(key=lambda x: (freq[x], x))
    return usable[:2]


def generate(s1_path, indexes, freq, output_path):
    exact_name, country_name, country_number, country_addr = indexes

    total = 0
    with_cand = 0
    total_pairs = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk in read_source(s1_path):
            for eid, country_raw, raw_name, raw_address in zip(
                chunk["entity_id"],
                chunk["country"],
                chunk["business_name"],
                chunk["business_address"],
            ):
                total += 1
                country = norm(country_raw)
                name = norm(raw_name)
                address = norm(raw_address)

                candidates = set()

                # 1. Exact normalized name.
                if name:
                    candidates.update(exact_name.get(name, ()))

                # 2. Bounded country + rare-name-token block.
                #    Only small postings are used, preventing runaway output.
                for tok in rare_tokens(name, freq):
                    candidates.update(country_name.get((country, tok), ()))

                # 3. Bounded address-number complement.
                number = addr_number(address)
                if number:
                    candidates.update(
                        country_number.get((country, number), ())
                    )

                # 4. Bounded address-token complement.
                for tok in set(tokens(address)):
                    candidates.update(
                        country_addr.get((country, tok), ())
                    )

                ids = sorted(candidates)

                if ids:
                    with_cand += 1
                    total_pairs += len(ids)

                out.write(f"{eid}\t{','.join(ids)}\n")

            # Visible progress so "stuck" is impossible to diagnose blindly.
            print(
                f"[gen] processed {total:,} S1 | "
                f"pairs {total_pairs:,}",
                flush=True,
            )

    print("\n" + "=" * 78)
    print("candidate_generator_v3_bounded COMPLETE")
    print("=" * 78)
    print(f"S1 entities          : {total:,}")
    print(f"S1 with candidates   : {with_cand:,}")
    print(f"Candidate coverage   : {100*with_cand/total:.4f}%")
    print(f"Total candidate pairs: {total_pairs:,}")
    print(f"Average / S1         : {total_pairs/total:.2f}")
    print(f"Output               : {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = time.time()
    paths = [args.source2, args.source3]

    print("=" * 78)
    print("candidate_generator_v3_bounded")
    print("=" * 78)

    print("\n[1/3] Counting name-token frequencies...")
    freq = count_name_tokens(paths)

    print("\n[2/3] Building bounded indexes...")
    indexes = build_indexes(paths, freq)

    print("\n[3/3] Generating validation candidates...")
    generate(args.s1, indexes, freq, args.output)

    print(f"\nTotal runtime: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
