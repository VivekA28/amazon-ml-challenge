from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK_SIZE = 100_000
MIN_TOKEN_LEN = 4
MAX_TOKEN_FREQ = 50_000

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\b\d+[a-z]?\b")


def norm(value):
    value = "" if value is None else str(value)
    value = value.lower().strip().replace("&", "and")
    value = _NON_ALNUM.sub(" ", value)
    return " ".join(value.split())


def name_tokens(value):
    return {x for x in value.split() if len(x) >= MIN_TOKEN_LEN}


def addr_number(value):
    m = _NUMBER.search(value)
    return m.group(0) if m else None


def read_source(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    )


def parse_ids(value):
    if not value:
        return []
    return [x for x in str(value).split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True, type=Path)
    ap.add_argument("--source2", required=True, type=Path)
    ap.add_argument("--source3", required=True, type=Path)
    ap.add_argument("--ground-truth", required=True, type=Path)
    args = ap.parse_args()

    t0 = time.time()

    # ============================================================
    # 1. GLOBAL NAME TOKEN FREQUENCIES
    # ============================================================
    print("[1/5] Counting S2+S3 name-token frequencies...")

    freq = Counter()

    for path in [args.source2, args.source3]:
        started = time.time()
        print(f"  {path.name}")

        for chunk in read_source(path):
            for raw in chunk["business_name"]:
                freq.update(name_tokens(norm(raw)))

        print(f"  done in {(time.time() - started) / 60:.1f} min")

    usable = {
        tok
        for tok, f in freq.items()
        if len(tok) >= MIN_TOKEN_LEN and f <= MAX_TOKEN_FREQ
    }

    print(f"Usable tokens: {len(usable):,}")

    # ============================================================
    # 2. BUILD EXACT V2 INDEXES
    # ============================================================
    print("[2/5] Building exact V2 blocking indexes...")

    exact_name = defaultdict(set)
    country_name = defaultdict(set)
    country_number = defaultdict(set)
    country_addr = defaultdict(set)

    for path in [args.source2, args.source3]:
        started = time.time()
        print(f"  {path.name}")

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

                # ----------------------------
                # Layer A index
                # ----------------------------
                exact_name[name].add(eid)

                # ----------------------------
                # Name-token index
                # ----------------------------
                for tok in name_tokens(name):
                    if tok in usable:
                        country_name[(country, tok)].add(eid)

                # ----------------------------
                # Address-number index
                # ----------------------------
                number = addr_number(address)

                if number:
                    country_number[(country, number)].add(eid)

                # ----------------------------
                # Address-token index
                # ----------------------------
                for tok in name_tokens(address):
                    posting = country_addr[(country, tok)]

                    # Same behavior as candidate_generator_top2.py
                    if len(posting) <= MAX_TOKEN_FREQ:
                        posting.add(eid)

        print(f"  done in {(time.time() - started) / 60:.1f} min")

    print(f"Exact-name keys       : {len(exact_name):,}")
    print(f"Country-name keys     : {len(country_name):,}")
    print(f"Country-number keys   : {len(country_number):,}")
    print(f"Country-address keys  : {len(country_addr):,}")

    # ============================================================
    # 3. LOAD GROUND TRUTH
    # ============================================================
    print("[3/5] Loading validation ground truth...")

    gt = pd.read_csv(
        args.ground_truth,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt_map = {}

    for _, row in gt.iterrows():
        gt_map[row["source1_entity_id"]] = set(
            parse_ids(row["matched_entity_ids"])
        )

    total_true = sum(len(x) for x in gt_map.values())

    # ============================================================
    # 4. EXACT V3 TOP-3 BLOCKING ESTIMATE
    # ============================================================
    print("[4/5] Estimating TOP-3 candidate volume...")

    counts = []

    total_candidates = 0
    with_candidates = 0
    recovered_true = 0
    s1_total = 0

    for chunk in read_source(args.s1):

        names = chunk["business_name"].map(norm)
        addrs = chunk["business_address"].map(norm)

        for eid, country_raw, name, address in zip(
            chunk["entity_id"],
            chunk["country"],
            names,
            addrs,
        ):
            s1_total += 1
            country = norm(country_raw)

            candidates = set()

            # ====================================================
            # Layer A: EXACT NORMALIZED NAME
            # ====================================================
            candidates.update(
                exact_name.get(name, ())
            )

            # ====================================================
            # Select TOP 3 rare usable name tokens
            # ====================================================
            usable_tokens = [
                tok
                for tok in name_tokens(name)
                if tok in usable
            ]

            usable_tokens.sort(
                key=lambda tok: (freq[tok], tok)
            )

            top3 = usable_tokens[:3]

            number = addr_number(address)

            # ====================================================
            # Layer B:
            # country + address number + top-3 name token
            # ====================================================
            if number:

                number_posting = country_number.get(
                    (country, number),
                    (),
                )

                if number_posting:

                    for tok in top3:

                        name_posting = country_name.get(
                            (country, tok),
                            (),
                        )

                        if not name_posting:
                            continue

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

            # ====================================================
            # Layer C:
            # country + address token + top-3 name token
            # ====================================================
            if address and top3:

                best_addr_posting = None

                for atok in name_tokens(address):

                    posting = country_addr.get(
                        (country, atok)
                    )

                    if posting and (
                        best_addr_posting is None
                        or len(posting) < len(best_addr_posting)
                    ):
                        best_addr_posting = posting

                if best_addr_posting:

                    for tok in top3:

                        name_posting = country_name.get(
                            (country, tok),
                            (),
                        )

                        if not name_posting:
                            continue

                        if len(name_posting) < len(best_addr_posting):

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

            # ====================================================
            # Statistics
            # ====================================================
            n = len(candidates)

            counts.append(n)
            total_candidates += n

            if n:
                with_candidates += 1

            if eid in gt_map:
                recovered_true += len(
                    candidates & gt_map[eid]
                )

    # ============================================================
    # 5. REPORT
    # ============================================================
    arr = np.asarray(counts, dtype=np.int64)

    print()
    print("=" * 78)
    print("TOP-3 BLOCKING ESTIMATE — EXACT V2 LOGIC")
    print("=" * 78)

    print(f"S1 entities              : {s1_total:,}")
    print(f"S1 with candidates       : {with_candidates:,}")

    print(
        f"S1 coverage              : "
        f"{100 * with_candidates / s1_total:.4f}%"
    )

    print(
        f"Estimated candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Average / S1             : "
        f"{total_candidates / s1_total:.2f}"
    )

    print()
    print("CANDIDATE COUNT DISTRIBUTION")
    print("-" * 78)

    for p in [50, 90, 95, 99, 99.9, 100]:

        print(
            f"P{p:<5}: "
            f"{np.percentile(arr, p):,.0f}"
        )

    print()
    print(f"True pairs               : {total_true:,}")
    print(f"Recovered true pairs     : {recovered_true:,}")

    print(
        f"Estimated blocking recall: "
        f"{100 * recovered_true / total_true:.4f}%"
    )

    print()
    print(
        f"Runtime: "
        f"{(time.time() - t0) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()