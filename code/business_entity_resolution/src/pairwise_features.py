#!/usr/bin/env python3
"""
Generate pairwise features for candidate pairs.

The lookup DB must contain BOTH:
  S1 records
  S2/S3 records

Build it with build_entity_lookup.py using:
  --source1 data/validation/val_source1.tsv
  --source2 data/train/train_source2.tsv
  --source3 data/train/train_source3.tsv

First sanity run:
  --max-s1 10000

Output is row-per-candidate:
  source1_entity_id, candidate_entity_id, features..., label
"""
from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio

_NUMBER_RE = re.compile(r"\b\d+[a-z]?\b")

FEATURE_NAMES = [
    "name_exact",
    "name_core_exact",
    "name_jaccard",
    "name_overlap",
    "name_ratio",
    "address_exact",
    "address_jaccard",
    "address_overlap",
    "address_ratio",
    "address_number_equal",
    "country_equal",
    "name_address_mean",
    "name_address_min",
]


def tokens(s: str) -> set[str]:
    return set(s.split()) if s else set()


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def number(s: str) -> str:
    m = _NUMBER_RE.search(s or "")
    return m.group(0) if m else ""


def make_features(a: dict, b: dict) -> list[float]:
    name_exact = float(bool(a["name_norm"]) and a["name_norm"] == b["name_norm"])
    name_core_exact = float(
        bool(a["name_core"]) and a["name_core"] == b["name_core"]
    )

    name_j = jaccard(a["name_norm"], b["name_norm"])
    name_o = overlap(a["name_norm"], b["name_norm"])
    name_r = ratio(a["name_norm"], b["name_norm"]) / 100.0

    addr_exact = float(
        bool(a["address_norm"]) and a["address_norm"] == b["address_norm"]
    )
    addr_j = jaccard(a["address_norm"], b["address_norm"])
    addr_o = overlap(a["address_norm"], b["address_norm"])
    addr_r = ratio(a["address_norm"], b["address_norm"]) / 100.0

    na, nb = number(a["address_norm"]), number(b["address_norm"])
    number_equal = float(bool(na) and bool(nb) and na == nb)

    country_equal = float(
        bool(a["country_norm"])
        and bool(b["country_norm"])
        and a["country_norm"] == b["country_norm"]
    )

    return [
        name_exact,
        name_core_exact,
        name_j,
        name_o,
        name_r,
        addr_exact,
        addr_j,
        addr_o,
        addr_r,
        number_equal,
        country_equal,
        (name_r + addr_r) / 2.0,
        min(name_r, addr_r),
    ]


def fetch_records(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    out = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        sql = f"""
        SELECT entity_id, name_norm, name_core, address_norm, country_norm
        FROM entities
        WHERE entity_id IN ({placeholders})
        """
        for row in conn.execute(sql, batch):
            out[row[0]] = {
                "name_norm": row[1] or "",
                "name_core": row[2] or "",
                "address_norm": row[3] or "",
                "country_norm": row[4] or "",
            }
    return out


def load_gt(path: Path, ids: set[str]) -> dict[str, set[str]]:
    gt = {}
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, chunksize=100_000, keep_default_na=False
    ):
        sub = chunk[chunk["source1_entity_id"].isin(ids)]
        for sid, mids in zip(sub["source1_entity_id"], sub["matched_entity_ids"]):
            gt[str(sid)] = set(x for x in str(mids).split(",") if x)
    return gt


def iter_candidate_chunks(path: Path, chunk_s1: int):
    # pandas chunking is by rows, which here equals S1 rows.
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunk_s1,
        keep_default_na=False,
    ):
        rows = []
        for sid, ids in zip(chunk["source1_entity_id"], chunk["candidate_entity_ids"]):
            cids = [x for x in str(ids).split(",") if x]
            rows.append((str(sid), cids))
        yield rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--lookup-db", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-s1", type=int, default=None)
    ap.add_argument("--chunk-s1", type=int, default=5000)
    args = ap.parse_args()

    # GT is one row per S1 and is small enough relative to the 24.4M-record
    # corpus to keep as a compact Python dict. It is loaded exactly once.
    # This avoids rescanning GT for every candidate chunk.
    gt = {}
    for chunk in pd.read_csv(
        args.ground_truth,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False,
    ):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            gt[str(sid)] = set(x for x in str(mids).split(",") if x)

    conn = sqlite3.connect(args.lookup_db)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["source1_entity_id", "candidate_entity_id", *FEATURE_NAMES, "label"]

    first = True
    total_s1 = 0
    total_pairs = 0
    missing = 0

    try:
        for rows in iter_candidate_chunks(args.candidates, args.chunk_s1):
            if args.max_s1 is not None and total_s1 >= args.max_s1:
                break

            if args.max_s1 is not None:
                remaining = args.max_s1 - total_s1
                rows = rows[:remaining]

            s1_ids = [sid for sid, _ in rows]
            candidate_ids = [cid for _, cids in rows for cid in cids]
            all_ids = list(set(s1_ids + candidate_ids))

            records = fetch_records(conn, all_ids)

            mode = "w" if first else "a"
            with args.output.open(mode, encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames, delimiter="\t"
                )
                if first:
                    writer.writeheader()

                for sid, cids in rows:
                    a = records.get(sid)
                    if a is None:
                        missing += len(cids)
                        continue

                    positives = gt.get(sid, set())

                    for cid in cids:
                        b = records.get(cid)
                        if b is None:
                            missing += 1
                            continue

                        feats = make_features(a, b)
                        writer.writerow(
                            {
                                "source1_entity_id": sid,
                                "candidate_entity_id": cid,
                                **dict(zip(FEATURE_NAMES, feats)),
                                "label": int(cid in positives),
                            }
                        )
                        total_pairs += 1

            first = False
            total_s1 += len(rows)

            print(
                f"[features] S1 {total_s1:,} | pairs {total_pairs:,} | "
                f"missing {missing:,}",
                flush=True,
            )

    finally:
        conn.close()

    print("\n" + "=" * 72)
    print("PAIRWISE FEATURE GENERATION COMPLETE")
    print("=" * 72)
    print(f"S1 processed:    {total_s1:,}")
    print(f"Candidate pairs: {total_pairs:,}")
    print(f"Missing records: {missing:,}")
    print(f"Output:          {args.output}")


if __name__ == "__main__":
    main()
