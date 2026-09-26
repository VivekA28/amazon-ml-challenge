#!/usr/bin/env python3
"""
Resume-only V2 ∪ F2 analysis.

IMPORTANT:
- Does NOT regenerate F2.
- Does NOT load V2's 18.8M candidate IDs into DuckDB.
- Does NOT perform large DuckDB joins.
- Streams V2 TSV and existing F2 edge tables.
- Uses the validation GT (~1.14M true pairs) as an in-memory lookup.

Existing DB:
  output/candidate_f2_partitioned.duckdb
"""

from pathlib import Path
import csv
import duckdb
import time

BASE = Path("/home/vivek/Desktop/amazon-ml-challenge")
DB_PATH = BASE / "output/candidate_f2_partitioned.duckdb"
V2_PATH = BASE / "output/candidate_pairs_validation_top2.tsv"
GT_PATH = BASE / "data/validation/val_ground_truth.tsv"

PARTITIONS = 16

def main():
    t0 = time.time()

    if not DB_PATH.exists():
        raise SystemExit(f"Missing DB: {DB_PATH}")
    if not V2_PATH.exists():
        raise SystemExit(f"Missing V2 file: {V2_PATH}")
    if not GT_PATH.exists():
        raise SystemExit(f"Missing GT file: {GT_PATH}")

    print("Resume V2 ∪ F2 analysis — streaming mode")
    print(f"DB: {DB_PATH}")

    # ---------------------------------------------------------------
    # 1. Load GT into a compact Python lookup.
    # ---------------------------------------------------------------
    print("[1/4] Loading validation GT into memory...")

    gt = {}
    gt_total = 0

    with open(GT_PATH, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        if "source1_entity_id" not in rd.fieldnames:
            raise RuntimeError(f"Unexpected GT columns: {rd.fieldnames}")

        for row in rd:
            sid = row["source1_entity_id"]
            ids = {
                x.strip()
                for x in row["matched_entity_ids"].split(",")
                if x.strip()
            }
            gt[sid] = ids
            gt_total += len(ids)

    print(f"    S1 GT rows: {len(gt):,}")
    print(f"    true pairs: {gt_total:,}")

    # ---------------------------------------------------------------
    # 2. Stream V2 candidate file.
    # ---------------------------------------------------------------
    print("[2/4] Streaming V2 candidates...")

    v2_hit_pairs = set()
    v2_candidates = 0

    with open(V2_PATH, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")

        expected = {"source1_entity_id", "candidate_entity_ids"}
        if not expected.issubset(set(rd.fieldnames or [])):
            raise RuntimeError(f"Unexpected V2 columns: {rd.fieldnames}")

        for row in rd:
            sid = row["source1_entity_id"]
            cand_text = row["candidate_entity_ids"] or ""
            if not cand_text:
                continue

            for cid in cand_text.split(","):
                cid = cid.strip()
                if not cid:
                    continue

                v2_candidates += 1

                true_ids = gt.get(sid)
                if true_ids is not None and cid in true_ids:
                    v2_hit_pairs.add((sid, cid))

    v2_hit = len(v2_hit_pairs)
    print(f"    V2 candidates: {v2_candidates:,}")
    print(f"    V2 recovered: {v2_hit:,}")
    print(f"    V2 recall: {100.0*v2_hit/gt_total:.4f}%")

    # ---------------------------------------------------------------
    # 3. Stream existing F2 edge tables directly from DuckDB.
    #
    # No JOIN/GROUP BY. The GT lookup is already in Python.
    # ---------------------------------------------------------------
    print("[3/4] Streaming existing F2 edge tables...")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='1GB'")

    f2_candidates = 0
    f2_hit_pairs = set()

    for p in range(PARTITIONS):
        table = f"edge_{p}"

        exists = con.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name='{table}'
        """).fetchone()[0]

        if not exists:
            raise RuntimeError(f"Missing {table} in existing DB.")

        print(f"    partition {p+1:02d}/{PARTITIONS}...", flush=True)

        cur = con.execute(f"""
            SELECT source1_id, candidate_id
            FROM {table}
        """)

        while True:
            rows = cur.fetchmany(100_000)
            if not rows:
                break

            f2_candidates += len(rows)

            for sid, cid in rows:
                sid = str(sid)
                cid = str(cid)
                true_ids = gt.get(sid)

                if true_ids is not None and cid in true_ids:
                    f2_hit_pairs.add((sid, cid))

    con.close()

    f2_hit = len(f2_hit_pairs)
    f2_only = len(f2_hit_pairs - v2_hit_pairs)
    union_hit = len(v2_hit_pairs | f2_hit_pairs)

    print(f"    F2 candidates: {f2_candidates:,}")
    print(f"    F2 recovered: {f2_hit:,}")
    print(f"    F2 recall: {100.0*f2_hit/gt_total:.4f}%")

    # ---------------------------------------------------------------
    # 4. Final results.
    # ---------------------------------------------------------------
    print("[4/4] FINAL RESULTS")
    print("=" * 62)
    print(f"GT true pairs:             {gt_total:,}")
    print(f"V2 candidates:             {v2_candidates:,}")
    print(f"F2 candidates:             {f2_candidates:,}")
    print(f"Naive V2 + F2:             {v2_candidates + f2_candidates:,}")
    print()
    print(f"V2 recovered:              {v2_hit:,}")
    print(f"V2 recall:                 {100.0*v2_hit/gt_total:.4f}%")
    print()
    print(f"F2 recovered:              {f2_hit:,}")
    print(f"F2 recall:                 {100.0*f2_hit/gt_total:.4f}%")
    print()
    print(f"F2 recovers V2 misses:     {f2_only:,}")
    print(f"UNION recovered:           {union_hit:,}")
    print(f"V2 + F2 union recall:      {100.0*union_hit/gt_total:.4f}%")
    print(f"Gain over V2:              {100.0*(union_hit-v2_hit)/gt_total:.4f} pp")
    print("=" * 62)
    print(f"Analysis runtime: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
