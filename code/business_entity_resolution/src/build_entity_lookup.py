#!/usr/bin/env python3
"""
Build a disk-backed SQLite lookup for Source-2/Source-3 records.

This avoids loading all ~10.3M S2/S3 rows into Python dictionaries.
The database stores only the fields needed by pairwise feature generation.

Run once after the canonical preprocessing step is available, OR directly
from raw TSVs for the first sanity run.

Example:
  python3 build_entity_lookup.py \
      --source2 data/train/train_source2.tsv \
      --source3 data/train/train_source3.tsv \
      --output data/feature_lookup.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

# Import the project's canonical normalization functions.
# Adjust this import only if this script is placed outside src/.
try:
    from preprocess import normalize_name, normalize_address, normalize_country
except ImportError:
    # If preprocess.py is in the same src directory, this is normally enough.
    raise SystemExit(
        "Could not import preprocess.py. Put this script beside preprocess.py "
        "or add that directory to PYTHONPATH."
    )

CHUNK_SIZE = 100_000

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    name_norm TEXT NOT NULL,
    name_core TEXT NOT NULL,
    address_norm TEXT NOT NULL,
    country_norm TEXT NOT NULL
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO entities
(entity_id, name_norm, name_core, address_norm, country_norm)
VALUES (?, ?, ?, ?, ?)
"""


def read_chunks(path: Path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    )


def process_file(conn: sqlite3.Connection, path: Path) -> int:
    total = 0
    start = time.time()

    for chunk in read_chunks(path):
        names = chunk["business_name"].fillna("").map(normalize_name)
        rows = []

        for eid, name_result, address, country in zip(
            chunk["entity_id"],
            names,
            chunk["business_address"].fillna(""),
            chunk["country"].fillna(""),
        ):
            name_norm, name_core, _suffix = name_result
            rows.append(
                (
                    str(eid),
                    name_norm,
                    name_core,
                    normalize_address(str(address)),
                    normalize_country(str(country)),
                )
            )

        conn.executemany(INSERT_SQL, rows)
        conn.commit()

        total += len(rows)
        print(
            f"[lookup] {path.name}: {total:,} rows "
            f"({(time.time() - start) / 60:.1f} min)",
            flush=True,
        )

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True, type=Path)
    ap.add_argument("--source2", required=True, type=Path)
    ap.add_argument("--source3", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.output)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(CREATE_SQL)
    conn.commit()

    start = time.time()
    try:
        n1 = process_file(conn, args.source1)
        n2 = process_file(conn, args.source2)
        n3 = process_file(conn, args.source3)

        # PRIMARY KEY already creates an index on entity_id.
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()

    print("\n" + "=" * 72)
    print("LOOKUP BUILD COMPLETE")
    print("=" * 72)
    print(f"S1 rows: {n1:,}")
    print(f"S2 rows: {n2:,}")
    print(f"S3 rows: {n3:,}")
    print(f"DB:      {args.output}")
    print(f"Time:    {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
