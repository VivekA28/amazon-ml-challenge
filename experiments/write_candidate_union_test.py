#!/usr/bin/env python3

from pathlib import Path
import duckdb
import time

BASE = Path("/home/vivek/Desktop/amazon-ml-challenge")
DB = BASE / "output/candidate_union_test.duckdb"
OUT = BASE / "output/candidate_pairs_test_union.tsv"
TMP = BASE / "output/candidate_union_test_tmp"

HASH_PARTS = 16
MEMORY = "8GB"


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()

    TMP.mkdir(parents=True, exist_ok=True)

    if OUT.exists():
        OUT.unlink()

    con = duckdb.connect(str(DB))

    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")

    total_edges = con.execute(
        "SELECT COUNT(*) FROM union_edges"
    ).fetchone()[0]

    total_s1 = con.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM union_edges"
    ).fetchone()[0]

    log("=" * 72)
    log("WRITING TEST CANDIDATE UNION")
    log("=" * 72)
    log(f"Union edges: {total_edges:,}")
    log(f"S1 with candidates: {total_s1:,}")
    log(f"Hash partitions: {HASH_PARTS}")

    part_files = []

    for hp in range(HASH_PARTS):
        part = TMP / f"union_part_{hp:02d}.tsv"
        part_files.append(part)

        if part.exists():
            part.unlink()

        log(f"[{hp + 1}/{HASH_PARTS}] Aggregating hash partition...")

        con.execute(f"""
            COPY (
                SELECT
                    source1_entity_id,
                    string_agg(
                        candidate_entity_id,
                        ',' ORDER BY candidate_entity_id
                    ) AS candidate_entity_ids
                FROM union_edges
                WHERE MOD(
                    ABS(hash(source1_entity_id)),
                    {HASH_PARTS}
                ) = {hp}
                GROUP BY source1_entity_id
                ORDER BY source1_entity_id
            )
            TO '{part.as_posix()}'
            (HEADER, DELIMITER '\\t')
        """)

        log(
            f"    wrote {part.stat().st_size / (1024**3):.2f} GB"
        )

    con.close()

    log("[merge] Combining partition files...")

    with open(OUT, "wb") as fout:
        fout.write(b"source1_entity_id\tcandidate_entity_ids\n")

        for part in part_files:
            log(f"    appending {part.name}")
            with open(part, "rb") as fin:
                # Skip partition header.
                fin.readline()
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)

    log("[cleanup] Removing temporary partition files...")

    for part in part_files:
        part.unlink(missing_ok=True)

    log("")
    log("=" * 72)
    log("TEST CANDIDATE UNION COMPLETE")
    log("=" * 72)
    log(f"S1 with candidates: {total_s1:,}")
    log(f"Union edges:        {total_edges:,}")
    log(f"Output:             {OUT}")
    log(f"Output size:        {OUT.stat().st_size / (1024**3):.2f} GB")
    log(f"Runtime:             {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
