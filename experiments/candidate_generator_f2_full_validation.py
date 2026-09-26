#!/usr/bin/env python3
"""
Full-validation F2 candidate generator — corrected, low-memory version.

F2 configuration:
  DF cap       = 0.001
  TOP_K        = 5 per entity
  bucket cap   = 6000 per token, GLOBAL
  min shared   = 2

Correctness requirements:
  - Reference TOP_K is applied AFTER the GLOBAL bucket cap.
  - S1 TOP_K is applied using the same global token DF.
  - Global bucket cap is NOT applied independently per partition.
  - V2 uses source1_entity_id / candidate_entity_ids.

This script is intended to be run from the project environment.
It writes:
  output/candidate_pairs_validation_f2.tsv
and retains the existing partitioned/low-memory strategy.
"""

from pathlib import Path
import csv
import heapq
import shutil
import time
import duckdb

BASE = Path("/home/vivek/Desktop/amazon-ml-challenge")
S1_PATH = BASE / "processed/validation/val_source1.parquet"
S2_PATH = BASE / "processed/validation/train_source2.parquet"
S3_PATH = BASE / "processed/validation/train_source3.parquet"
GT_PATH = BASE / "data/validation/val_ground_truth.tsv"

OUT_DIR = BASE / "output"
WORK_DIR = OUT_DIR / "f2_corrected_work"
DB_PATH = OUT_DIR / "candidate_f2_corrected.duckdb"
OUT_PATH = OUT_DIR / "candidate_pairs_validation_f2.tsv"

PARTITIONS = 16
DF_CAP = 0.001
TOP_K = 5
BUCKET_CAP = 6000
MIN_SHARED = 2
MEMORY = "9GB"

def q(con, sql):
    return con.execute(sql)

def log(msg):
    print(msg, flush=True)

def reset():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    if OUT_PATH.exists():
        OUT_PATH.unlink()

def main():
    t0 = time.time()
    reset()

    con = duckdb.connect(str(DB_PATH))
    q(con, f"SET memory_limit='{MEMORY}'")
    q(con, "SET threads=1")
    q(con, "SET preserve_insertion_order=false")
    q(con, f"SET temp_directory='{(WORK_DIR / 'duckdb_tmp').as_posix()}'")
    q(con, "SET enable_progress_bar=false")

    log("Corrected F2 full validation")
    log(
        f"partitions={PARTITIONS}, df_cap={DF_CAP}, top_k={TOP_K}, "
        f"bucket_cap={BUCKET_CAP}, min_shared={MIN_SHARED}"
    )

    # ---------------------------------------------------------------
    # 1. Reference base
    # ---------------------------------------------------------------
    log("[1/9] Loading reference address fields...")
    q(con, f"""
        CREATE TABLE ref AS
        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(country_norm AS VARCHAR) AS country,
            CAST(business_address_norm AS VARCHAR) AS address
        FROM read_parquet([
            '{S2_PATH.as_posix()}',
            '{S3_PATH.as_posix()}'
        ])
        WHERE business_address_norm IS NOT NULL
          AND business_address_norm <> ''
    """)
    ref_n = q(con, "SELECT COUNT(*) FROM ref").fetchone()[0]
    log(f"    reference rows: {ref_n:,}")

    # ---------------------------------------------------------------
    # 2. Reference entity partitions
    # ---------------------------------------------------------------
    log("[2/9] Partitioning reference rows...")
    for p in range(PARTITIONS):
        q(con, f"""
            CREATE TABLE ref_{p} AS
            SELECT entity_id, country, address
            FROM ref
            WHERE MOD(hash(entity_id), {PARTITIONS}) = {p}
        """)
    q(con, "DROP TABLE ref")
    q(con, "CHECKPOINT")

    # ---------------------------------------------------------------
    # 3. Tokenize partitions + global token DF
    # ---------------------------------------------------------------
    log("[3/9] Building reference token postings + global DF...")
    for p in range(PARTITIONS):
        t = time.time()

        q(con, f"""
            CREATE TABLE tok_{p} AS
            SELECT DISTINCT
                entity_id,
                country,
                lower(tok) AS token
            FROM ref_{p},
            LATERAL unnest(string_split(address, ' ')) AS u(tok)
            WHERE tok <> ''
        """)

        q(con, f"""
            CREATE TABLE df_{p} AS
            SELECT token, COUNT(*)::BIGINT AS df
            FROM tok_{p}
            GROUP BY token
        """)

        tok_n = q(con, f"SELECT COUNT(*) FROM tok_{p}").fetchone()[0]
        log(
            f"    partition {p+1:02d}/{PARTITIONS}: "
            f"{tok_n:,} unique entity-token rows ({time.time()-t:.1f}s)"
        )

    # Global DF. This is retained until BOTH reference and S1 TOP-K are done.
    union_df = "\nUNION ALL\n".join(
        f"SELECT * FROM df_{p}" for p in range(PARTITIONS)
    )
    q(con, f"""
        CREATE TABLE global_df AS
        SELECT token, SUM(df)::BIGINT AS df
        FROM ({union_df})
        GROUP BY token
    """)

    unique_tokens = q(con, "SELECT COUNT(*) FROM global_df").fetchone()[0]
    log(f"    global unique address tokens: {unique_tokens:,}")

    # ---------------------------------------------------------------
    # 4. GLOBAL DF filter -> GLOBAL bucket cap -> reference TOP-K
    # ---------------------------------------------------------------
    log("[4/9] Applying GLOBAL DF cap + GLOBAL bucket cap + reference TOP-K...")

    df_limit = max(1, int(ref_n * DF_CAP))
    log(f"    DF limit: {df_limit:,}")

    q(con, f"""
        CREATE TABLE surviving_tokens AS
        SELECT token, df
        FROM global_df
        WHERE df <= {df_limit}
    """)
    surv_n = q(con, "SELECT COUNT(*) FROM surviving_tokens").fetchone()[0]
    log(f"    tokens after DF cap: {surv_n:,}")

    # Filter each partition by the GLOBAL DF table.
    for p in range(PARTITIONS):
        q(con, f"""
            CREATE TABLE filtered_{p} AS
            SELECT t.entity_id, t.country, t.token
            FROM tok_{p} t
            JOIN surviving_tokens s USING (token)
        """)
        q(con, f"DROP TABLE tok_{p}")
        q(con, f"DROP TABLE df_{p}")

    # IMPORTANT:
    # Bucket cap is applied once over the UNION of ALL partitions.
    # Therefore one token can have at most 6000 reference postings globally.
    filtered_union = "\nUNION ALL\n".join(
        f"SELECT * FROM filtered_{p}" for p in range(PARTITIONS)
    )

    HASH_BUCKETS = 8

    for hb in range(HASH_BUCKETS):
        q(con, f"""
            CREATE TABLE ref_bucketed_{hb} AS
            SELECT entity_id, country, token
            FROM (
                SELECT
                    entity_id,
                    country,
                    token,
                    ROW_NUMBER() OVER (
                        PARTITION BY token
                        ORDER BY entity_id
                    ) AS token_rank
                FROM ({filtered_union})
                WHERE MOD(hash(token), {HASH_BUCKETS}) = {hb}
            )
            WHERE token_rank <= {BUCKET_CAP}
        """)
        log(f"    hash bucket {hb+1}/{HASH_BUCKETS} done")

    ref_bucketed_union = "\nUNION ALL\n".join(
        f"SELECT * FROM ref_bucketed_{hb}" for hb in range(HASH_BUCKETS)
    )

    q(con, f"""
        CREATE TABLE ref_bucketed AS
        SELECT * FROM ({ref_bucketed_union})
    """)

    for hb in range(HASH_BUCKETS):
        q(con, f"DROP TABLE ref_bucketed_{hb}")
    for p in range(PARTITIONS):
        q(con, f"DROP TABLE filtered_{p}")

    bucketed_n = q(con, "SELECT COUNT(*) FROM ref_bucketed").fetchone()[0]
    log(f"    reference postings after GLOBAL bucket cap: {bucketed_n:,}")

    # Reference TOP-K: rarest surviving/bucketed tokens per entity.
    q(con, f"""
        CREATE TABLE ref_topk AS
        SELECT entity_id, country, token
        FROM (
            SELECT
                r.entity_id,
                r.country,
                r.token,
                g.df,
                ROW_NUMBER() OVER (
                    PARTITION BY r.entity_id
                    ORDER BY g.df ASC, r.token ASC
                ) AS entity_rank
            FROM ref_bucketed r
            JOIN global_df g USING (token)
        )
        WHERE entity_rank <= {TOP_K}
    """)

    ref_topk_n = q(con, "SELECT COUNT(*) FROM ref_topk").fetchone()[0]
    log(f"    reference postings after TOP_K: {ref_topk_n:,}")

    # ---------------------------------------------------------------
    # 5. Load S1 + apply S1 TOP-K using SAME global DF
    # ---------------------------------------------------------------
    log("[5/9] Loading S1 + applying S1 TOP-K...")
    q(con, f"""
        CREATE TABLE s1 AS
        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(country_norm AS VARCHAR) AS country,
            CAST(business_address_norm AS VARCHAR) AS address
        FROM read_parquet('{S1_PATH.as_posix()}')
    """)
    s1_n = q(con, "SELECT COUNT(*) FROM s1").fetchone()[0]
    log(f"    S1 rows: {s1_n:,}")

    q(con, """
        CREATE TABLE s1_tok_all AS
        SELECT DISTINCT
            entity_id,
            country,
            lower(tok) AS token
        FROM s1,
        LATERAL unnest(string_split(address, ' ')) AS u(tok)
        WHERE tok <> ''
    """)

    q(con, f"""
        CREATE TABLE s1_tok AS
        SELECT entity_id, country, token
        FROM (
            SELECT
                t.entity_id,
                t.country,
                t.token,
                g.df,
                ROW_NUMBER() OVER (
                    PARTITION BY t.entity_id
                    ORDER BY g.df ASC, t.token ASC
                ) AS entity_rank
            FROM s1_tok_all t
            JOIN global_df g USING (token)
            WHERE g.df <= {df_limit}
        )
        WHERE entity_rank <= {TOP_K}
    """)

    s1_tok_n = q(con, "SELECT COUNT(*) FROM s1_tok").fetchone()[0]
    log(f"    S1 TOP-K token rows: {s1_tok_n:,}")

    q(con, "DROP TABLE s1_tok_all")
    q(con, "DROP TABLE global_df")
    q(con, "DROP TABLE surviving_tokens")

    # ---------------------------------------------------------------
    # 6. Partition corrected reference postings + generate candidates
    # ---------------------------------------------------------------
    log("[6/9] Generating F2 candidates...")
    for p in range(PARTITIONS):
        q(con, f"""
            CREATE TABLE refp_{p} AS
            SELECT entity_id, country, token
            FROM ref_topk
            WHERE MOD(hash(entity_id), {PARTITIONS}) = {p}
        """)
    q(con, "DROP TABLE ref_topk")
    q(con, "CHECKPOINT")

    for p in range(PARTITIONS):
        t = time.time()
        q(con, f"""
            CREATE TABLE edge_{p} AS
            SELECT
                s.entity_id AS source1_id,
                r.entity_id AS candidate_id
            FROM s1_tok s
            JOIN refp_{p} r
              ON s.country = r.country
             AND s.token = r.token
            GROUP BY s.entity_id, r.entity_id
            HAVING COUNT(*) >= {MIN_SHARED}
        """)
        n = q(con, f"SELECT COUNT(*) FROM edge_{p}").fetchone()[0]
        q(con, f"DROP TABLE refp_{p}")
        log(
            f"    partition {p+1:02d}/{PARTITIONS}: "
            f"{n:,} candidate edges ({time.time()-t:.1f}s)"
        )

    # ---------------------------------------------------------------
    # 7. Streaming output
    # ---------------------------------------------------------------
    log("[7/9] Writing final candidate_pairs TSV (streaming merge)...")

    part_files = []
    for p in range(PARTITIONS):
        pf = WORK_DIR / f"candidate_part_{p:02d}.tsv"
        q(con, f"""
            COPY (
                SELECT
                    source1_id,
                    STRING_AGG(candidate_id, ',' ORDER BY candidate_id) AS candidate_entity_ids
                FROM edge_{p}
                GROUP BY source1_id
                ORDER BY source1_id
            )
            TO '{pf.as_posix()}'
            (HEADER, DELIMITER '\t')
        """)
        part_files.append(pf)

    s1_ids_path = WORK_DIR / "s1_ids.tsv"
    q(con, f"""
        COPY (
            SELECT entity_id
            FROM s1
            ORDER BY entity_id
        )
        TO '{s1_ids_path.as_posix()}'
        (HEADER, DELIMITER '\t')
    """)

    handles = []
    readers = []
    heap = []

    try:
        for idx, pf in enumerate(part_files):
            fh = open(pf, "r", encoding="utf-8", newline="")
            rd = csv.reader(fh, delimiter="\t")
            next(rd, None)
            handles.append(fh)
            readers.append(rd)
            try:
                row = next(rd)
                heapq.heappush(heap, (row[0], idx, row[1]))
            except StopIteration:
                pass

        s1_fh = open(s1_ids_path, "r", encoding="utf-8", newline="")
        s1_rd = csv.reader(s1_fh, delimiter="\t")
        next(s1_rd, None)

        with open(OUT_PATH, "w", encoding="utf-8", newline="") as out_fh:
            writer = csv.writer(out_fh, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

            for srow in s1_rd:
                source1_id = srow[0]
                chunks = []

                while heap and heap[0][0] < source1_id:
                    _, idx, _ = heapq.heappop(heap)
                    try:
                        nr = next(readers[idx])
                        heapq.heappush(heap, (nr[0], idx, nr[1]))
                    except StopIteration:
                        pass

                while heap and heap[0][0] == source1_id:
                    _, idx, cand = heapq.heappop(heap)
                    if cand:
                        chunks.append(cand)
                    try:
                        nr = next(readers[idx])
                        heapq.heappush(heap, (nr[0], idx, nr[1]))
                    except StopIteration:
                        pass

                writer.writerow([source1_id, ",".join(chunks)])

        s1_fh.close()
    finally:
        for fh in handles:
            fh.close()

    log(f"    wrote: {OUT_PATH}")

    # ---------------------------------------------------------------
    # 8. F2 recall — streaming GT lookup against existing edge tables
    # ---------------------------------------------------------------
    log("[8/9] Measuring F2 blocking recall...")

    gt = {}
    gt_total = 0
    with open(GT_PATH, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            sid = row["source1_entity_id"]
            ids = {x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()}
            gt[sid] = ids
            gt_total += len(ids)

    f2_hit_pairs = set()
    total_edges = 0

    for p in range(PARTITIONS):
        n = q(con, f"SELECT COUNT(*) FROM edge_{p}").fetchone()[0]
        total_edges += n
        cur = con.execute(f"SELECT source1_id, candidate_id FROM edge_{p}")
        while True:
            rows = cur.fetchmany(100_000)
            if not rows:
                break
            for sid, cid in rows:
                sid = str(sid)
                cid = str(cid)
                if cid in gt.get(sid, ()):
                    f2_hit_pairs.add((sid, cid))

    f2_hit = len(f2_hit_pairs)
    coverage_sids = set(f2_sid for f2_sid, _ in f2_hit_pairs)
    # Actual candidate coverage, not hit coverage:
    f2_candidate_sids = set()
    for p in range(PARTITIONS):
        cur = con.execute(f"SELECT DISTINCT source1_id FROM edge_{p}")
        while True:
            rows = cur.fetchmany(100_000)
            if not rows:
                break
            f2_candidate_sids.update(str(r[0]) for r in rows)

    coverage = 100.0 * len(f2_candidate_sids) / s1_n if s1_n else 0.0
    recall = 100.0 * f2_hit / gt_total if gt_total else 0.0

    log(f"    candidate edges: {total_edges:,}")
    log(f"    S1 with candidates: {len(f2_candidate_sids):,}")
    log(f"    coverage: {coverage:.4f}%")
    log(f"    true pairs: {gt_total:,}")
    log(f"    recovered: {f2_hit:,}")
    log(f"    blocking recall: {recall:.4f}%")

    # ---------------------------------------------------------------
    # 9. V2 ∪ F2 — stream V2 and compare with GT/F2 hit pairs
    # ---------------------------------------------------------------
    log("[9/9] Measuring V2 ∪ F2 union...")

    v2_path = OUT_DIR / "candidate_pairs_validation_top2.tsv"
    v2_hit_pairs = set()
    v2_candidates = 0

    with open(v2_path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        if rd.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            log(f"    V2 columns: {rd.fieldnames}")
        for row in rd:
            sid = row["source1_entity_id"]
            text = row["candidate_entity_ids"] or ""
            for cid in text.split(","):
                cid = cid.strip()
                if not cid:
                    continue
                v2_candidates += 1
                if cid in gt.get(sid, ()):
                    v2_hit_pairs.add((sid, cid))

    union_hit = len(v2_hit_pairs | f2_hit_pairs)
    f2_only = len(f2_hit_pairs - v2_hit_pairs)
    v2_hit = len(v2_hit_pairs)

    log(f"    V2 candidates: {v2_candidates:,}")
    log(f"    V2 recovered: {v2_hit:,} ({100*v2_hit/gt_total:.4f}%)")
    log(f"    F2 recovered: {f2_hit:,} ({100*f2_hit/gt_total:.4f}%)")
    log(f"    F2 recovers V2 misses: {f2_only:,}")
    log(f"    union recovered: {union_hit:,} ({100*union_hit/gt_total:.4f}%)")
    log(f"    union gain over V2: {100*(union_hit-v2_hit)/gt_total:.4f} pp")
    log(f"    naive V2+F2 candidates: {v2_candidates+total_edges:,}")
    log(f"DONE — total runtime: {(time.time()-t0)/60:.1f} min")

    con.close()

if __name__ == "__main__":
    main()
