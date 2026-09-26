#!/usr/bin/env python3

from pathlib import Path
import duckdb
import time

BASE = Path("/home/vivek/Desktop/amazon-ml-challenge")

V2 = BASE / "output/candidate_pairs_test_top2.tsv"
F2 = BASE / "output/candidate_pairs_test_f2.tsv"
OUT = BASE / "output/candidate_pairs_test_union.tsv"
DB = BASE / "output/candidate_union_test.duckdb"
TMP = BASE / "output/candidate_union_test_tmp"

PARTITIONS = 16
MEMORY = "8GB"

def log(msg):
    print(msg, flush=True)

def main():
    t0 = time.time()

    TMP.mkdir(parents=True, exist_ok=True)

    if DB.exists():
        DB.unlink()
    if OUT.exists():
        OUT.unlink()

    con = duckdb.connect(str(DB))

    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET enable_progress_bar=false")

    log("[1/4] Loading V2 + F2...")
    con.execute(f"""
        CREATE TABLE v2 AS
        SELECT
            source1_entity_id,
            candidate_entity_ids
        FROM read_csv(
            '{V2.as_posix()}',
            delim='\\t',
            header=true,
            columns={{
                'source1_entity_id':'VARCHAR',
                'candidate_entity_ids':'VARCHAR'
            }}
        )
    """)

    con.execute(f"""
        CREATE TABLE f2 AS
        SELECT
            source1_entity_id,
            candidate_entity_ids
        FROM read_csv(
            '{F2.as_posix()}',
            delim='\\t',
            header=true,
            columns={{
                'source1_entity_id':'VARCHAR',
                'candidate_entity_ids':'VARCHAR'
            }}
        )
    """)

    v2_rows = con.execute("SELECT COUNT(*) FROM v2").fetchone()[0]
    f2_rows = con.execute("SELECT COUNT(*) FROM f2").fetchone()[0]

    log(f"    V2 S1 rows: {v2_rows:,}")
    log(f"    F2 S1 rows: {f2_rows:,}")

    log("[2/4] Exploding candidate lists...")
    con.execute("""
        CREATE TABLE v2_edges AS
        SELECT
            source1_entity_id,
            unnest(string_split(candidate_entity_ids, ',')) AS candidate_entity_id
        FROM v2
        WHERE candidate_entity_ids <> ''
    """)

    con.execute("""
        CREATE TABLE f2_edges AS
        SELECT
            source1_entity_id,
            unnest(string_split(candidate_entity_ids, ',')) AS candidate_entity_id
        FROM f2
        WHERE candidate_entity_ids <> ''
    """)

    v2_edges = con.execute("SELECT COUNT(*) FROM v2_edges").fetchone()[0]
    f2_edges = con.execute("SELECT COUNT(*) FROM f2_edges").fetchone()[0]

    log(f"    V2 edges: {v2_edges:,}")
    log(f"    F2 edges: {f2_edges:,}")

    log("[3/4] Building deduplicated union...")

    con.execute("""
        CREATE TABLE union_edges AS
        SELECT source1_entity_id, candidate_entity_id
        FROM v2_edges

        UNION

        SELECT source1_entity_id, candidate_entity_id
        FROM f2_edges
    """)

    union_edges = con.execute(
        "SELECT COUNT(*) FROM union_edges"
    ).fetchone()[0]

    log(f"    Union edges: {union_edges:,}")
    log(f"    Duplicates removed: {v2_edges + f2_edges - union_edges:,}")

    log("[4/4] Writing final union...")

    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                string_agg(
                    candidate_entity_id,
                    ',' ORDER BY candidate_entity_id
                ) AS candidate_entity_ids
            FROM union_edges
            GROUP BY source1_entity_id
            ORDER BY source1_entity_id
        )
        TO '{OUT.as_posix()}'
        (HEADER, DELIMITER '\\t')
    """)

    s1_count = con.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM union_edges"
    ).fetchone()[0]

    max_candidates = con.execute("""
        SELECT MAX(cnt)
        FROM (
            SELECT source1_entity_id, COUNT(*) AS cnt
            FROM union_edges
            GROUP BY source1_entity_id
        )
    """).fetchone()[0]

    log("")
    log("=" * 72)
    log("TEST CANDIDATE UNION COMPLETE")
    log("=" * 72)
    log(f"S1 with candidates: {s1_count:,}")
    log(f"V2 edges:          {v2_edges:,}")
    log(f"F2 edges:          {f2_edges:,}")
    log(f"Union edges:       {union_edges:,}")
    log(f"Duplicates removed: {v2_edges + f2_edges - union_edges:,}")
    log(f"Max candidates/S1: {max_candidates:,}")
    log(f"Output:            {OUT}")
    log(f"Runtime:           {(time.time() - t0) / 60:.1f} min")

    con.close()

if __name__ == "__main__":
    main()
