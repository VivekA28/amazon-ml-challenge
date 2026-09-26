#!/usr/bin/env python3

import os
import time
import duckdb


# ============================================================
# CONFIG
# ============================================================

BASE = "/home/vivek/Desktop/amazon-ml-challenge"

S1_PARQUET = f"{BASE}/processed/validation/val_source1.parquet"
S2_PARQUET = f"{BASE}/processed/validation/train_source2.parquet"
S3_PARQUET = f"{BASE}/processed/validation/train_source3.parquet"

GT_FILE = f"{BASE}/data/validation/val_ground_truth.tsv"

DB_FILE = f"{BASE}/output/address_blocking.duckdb"
TMP_DIR = f"{BASE}/output/duckdb_tmp"

PILOT_S1 = 75_000

CONFIGS = [
    {
        "name": "A",
        "df_fraction": 0.0005,   # 0.05%
        "top_k": 3,
        "bucket_cap": 3000,
    },
    {
        "name": "B",
        "df_fraction": 0.0010,   # 0.10%
        "top_k": 4,
        "bucket_cap": 5000,
    },
    {
        "name": "C",
        "df_fraction": 0.0005,   # 0.05%
        "top_k": 5,
        "bucket_cap": 3000,
    },
]


# ============================================================
# HELPERS
# ============================================================

def timer():
    return time.perf_counter()


def mins(start):
    return (time.perf_counter() - start) / 60.0


def sql_string(value):
    """
    Quote a SQL string literal.
    Example:
        abc.tsv -> 'abc.tsv'
    """
    return "'" + str(value).replace("'", "''") + "'"


def sql_identifier(value):
    """
    Quote a SQL identifier.
    Example:
        source1_entity_id -> "source1_entity_id"
    """
    return '"' + str(value).replace('"', '""') + '"'


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    con = duckdb.connect(DB_FILE)

    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(
        f"SET temp_directory={sql_string(TMP_DIR)}"
    )

    print("=" * 70)
    print("ADDRESS TOKEN PURGE PILOT")
    print("=" * 70)

    print(f"DuckDB:        {duckdb.__version__}")
    print("Memory limit:  4GB")
    print("Threads:       4")
    print(f"Temp:          {TMP_DIR}")
    print()

    # ========================================================
    # REFERENCE DATA
    # ========================================================

    start = timer()

    con.execute("DROP VIEW IF EXISTS reference")

    con.execute(f"""
        CREATE VIEW reference AS

        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(country_norm AS VARCHAR) AS country,
            CAST(business_address_norm AS VARCHAR) AS address

        FROM read_parquet(
            {sql_string(S2_PARQUET)}
        )

        UNION ALL

        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(country_norm AS VARCHAR) AS country,
            CAST(business_address_norm AS VARCHAR) AS address

        FROM read_parquet(
            {sql_string(S3_PARQUET)}
        )
    """)

    reference_rows = con.execute("""
        SELECT COUNT(*)
        FROM reference
    """).fetchone()[0]

    print(f"Reference records: {reference_rows:,}")

    # ========================================================
    # ADDRESS TOKENS
    # ========================================================

    con.execute("DROP TABLE IF EXISTS address_tokens")

    con.execute("""
        CREATE TABLE address_tokens AS

        SELECT DISTINCT
            r.entity_id,
            r.country,
            LOWER(TRIM(t.token)) AS token

        FROM reference AS r

        CROSS JOIN UNNEST(
            regexp_split_to_array(
                COALESCE(r.address, ''),
                '[^[:alnum:]]+'
            )
        ) AS t(token)

        WHERE
            TRIM(t.token) <> ''
            AND LENGTH(TRIM(t.token)) >= 2
    """)

    token_rows = con.execute("""
        SELECT COUNT(*)
        FROM address_tokens
    """).fetchone()[0]

    print(f"Address-token rows: {token_rows:,}")
    print(f"Time: {mins(start):.1f} min")

    # ========================================================
    # TOKEN DOCUMENT FREQUENCY
    # ========================================================

    start = timer()

    con.execute("DROP TABLE IF EXISTS token_df")

    con.execute("""
        CREATE TABLE token_df AS

        SELECT
            token,
            COUNT(*) AS token_df

        FROM address_tokens

        GROUP BY token
    """)

    unique_tokens = con.execute("""
        SELECT COUNT(*)
        FROM token_df
    """).fetchone()[0]

    print(f"Unique tokens: {unique_tokens:,}")
    print(f"Time: {mins(start):.1f} min")

    # ========================================================
    # LOAD PILOT S1
    # ========================================================

    con.execute("DROP TABLE IF EXISTS pilot_s1")

    con.execute(f"""
        CREATE TABLE pilot_s1 AS

        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(country_norm AS VARCHAR) AS country,
            CAST(business_address_norm AS VARCHAR) AS address

        FROM read_parquet(
            {sql_string(S1_PARQUET)}
        )

        LIMIT {PILOT_S1}
    """)

    pilot_rows = con.execute("""
        SELECT COUNT(*)
        FROM pilot_s1
    """).fetchone()[0]

    print(f"S1 pilot rows: {pilot_rows:,}")
    print()

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    start = timer()

    con.execute("DROP TABLE IF EXISTS gt_raw")

    con.execute(f"""
        CREATE TABLE gt_raw AS

        SELECT *
        FROM read_csv(
            {sql_string(GT_FILE)},
            delim='\\t',
            header=true,
            all_varchar=true,
            ignore_errors=true
        )
    """)

    gt_columns = con.execute("""
        SELECT column_name

        FROM information_schema.columns

        WHERE table_name = 'gt_raw'

        ORDER BY ordinal_position
    """).fetchall()

    gt_columns = [x[0] for x in gt_columns]

    print("Ground-truth columns:")

    for col in gt_columns:
        print(f"  {col}")

    # --------------------------------------------------------
    # Find S1 column
    # --------------------------------------------------------

    if "source1_entity_id" in gt_columns:
        s1_col = "source1_entity_id"
    else:
        raise RuntimeError(
            "Could not find source1_entity_id in GT columns: "
            f"{gt_columns}"
        )

    # --------------------------------------------------------
    # Find matched IDs column
    # --------------------------------------------------------

    match_candidates = [
        "matched_entity_ids",
        "matched_ids",
        "candidate_entity_ids",
    ]

    match_col = None

    for candidate in match_candidates:
        if candidate in gt_columns:
            match_col = candidate
            break

    if match_col is None:
        raise RuntimeError(
            "Could not find matched entity column. "
            f"GT columns: {gt_columns}"
        )

    print(f"S1 column: {s1_col}")
    print(f"Potential match columns: {[match_col]}")

    # ========================================================
    # PARSE GROUND TRUTH
    # ========================================================

    con.execute("DROP TABLE IF EXISTS gt")

    # IMPORTANT:
    # s1_col and match_col are SQL IDENTIFIERS.
    # Therefore they MUST use double quotes, not single quotes.

    s1_identifier = sql_identifier(s1_col)
    match_identifier = sql_identifier(match_col)

    con.execute(f"""
        CREATE TABLE gt AS

        SELECT DISTINCT
            CAST(
                g.{s1_identifier}
                AS VARCHAR
            ) AS source1_entity_id,

            TRIM(
                UNNEST(
                    regexp_split_to_array(
                        COALESCE(
                            g.{match_identifier},
                            ''
                        ),
                        '[,;]'
                    )
                )
            ) AS candidate_entity_id

        FROM gt_raw AS g

        WHERE
            COALESCE(
                TRIM(g.{match_identifier}),
                ''
            ) <> ''
    """)

    gt_rows = con.execute("""
        SELECT COUNT(*)
        FROM gt
    """).fetchone()[0]

    pilot_gt_rows = con.execute("""
        SELECT COUNT(DISTINCT g.source1_entity_id)

        FROM gt AS g

        INNER JOIN pilot_s1 AS s
            ON g.source1_entity_id = s.entity_id
    """).fetchone()[0]

    pilot_true_pairs = con.execute("""
        SELECT COUNT(*)

        FROM gt AS g

        INNER JOIN pilot_s1 AS s
            ON g.source1_entity_id = s.entity_id
    """).fetchone()[0]

    print(f"GT rows scanned: {gt_rows:,}")
    print(f"Pilot S1 with GT: {pilot_gt_rows:,}")
    print(f"Pilot true pairs: {pilot_true_pairs:,}")
    print(f"GT preparation time: {mins(start):.1f} min")
    print()

    # ========================================================
    # RUN CONFIGURATIONS
    # ========================================================

    results = []

    for cfg in CONFIGS:

        name = cfg["name"]
        df_fraction = cfg["df_fraction"]
        top_k = cfg["top_k"]
        bucket_cap = cfg["bucket_cap"]

        print("=" * 70)
        print(f"CONFIG {name}")
        print("=" * 70)

        print(f"DF fraction: {df_fraction}")
        print(f"Top-K:       {top_k}")
        print(f"Bucket cap:  {bucket_cap}")

        start_cfg = timer()

        # ====================================================
        # DF CAP
        # ====================================================

        df_cap = int(reference_rows * df_fraction)

        print(f"DF cap: {df_cap:,}")

        # ====================================================
        # GOOD TOKENS
        # ====================================================

        con.execute("DROP TABLE IF EXISTS good_tokens")

        con.execute(f"""
            CREATE TEMP TABLE good_tokens AS

            SELECT
                token,
                token_df

            FROM token_df

            WHERE token_df <= {df_cap}
        """)

        good_token_count = con.execute("""
            SELECT COUNT(*)
            FROM good_tokens
        """).fetchone()[0]

        print(
            f"Tokens after DF filter: "
            f"{good_token_count:,}"
        )

        # ====================================================
        # FILTER POSTINGS
        # ====================================================

        con.execute("DROP TABLE IF EXISTS postings")

        con.execute("""
            CREATE TEMP TABLE postings AS

            SELECT
                a.entity_id,
                a.country,
                a.token,
                g.token_df

            FROM address_tokens AS a

            INNER JOIN good_tokens AS g
                ON a.token = g.token
        """)

        postings_count = con.execute("""
            SELECT COUNT(*)
            FROM postings
        """).fetchone()[0]

        print(
            f"Postings after DF filter: "
            f"{postings_count:,}"
        )

        # ====================================================
        # BUCKET SIZES
        # ====================================================

        con.execute("DROP TABLE IF EXISTS bucket_sizes")

        con.execute("""
            CREATE TEMP TABLE bucket_sizes AS

            SELECT
                token,
                COUNT(*) AS bucket_size

            FROM postings

            GROUP BY token
        """)

        # ====================================================
        # PURGE LARGE BUCKETS
        # ====================================================

        con.execute("DROP TABLE IF EXISTS good_bucket_tokens")

        con.execute(f"""
            CREATE TEMP TABLE good_bucket_tokens AS

            SELECT
                token

            FROM bucket_sizes

            WHERE bucket_size <= {bucket_cap}
        """)

        buckets_after_purge = con.execute("""
            SELECT COUNT(*)
            FROM good_bucket_tokens
        """).fetchone()[0]

        print(
            f"Buckets after size purge: "
            f"{buckets_after_purge:,}"
        )

        # ====================================================
        # POSTINGS AFTER BUCKET PURGE
        # ====================================================

        con.execute("DROP TABLE IF EXISTS postings_purged")

        con.execute("""
            CREATE TEMP TABLE postings_purged AS

            SELECT
                p.entity_id,
                p.country,
                p.token,
                p.token_df

            FROM postings AS p

            INNER JOIN good_bucket_tokens AS b
                ON p.token = b.token
        """)

        purged_postings = con.execute("""
            SELECT COUNT(*)
            FROM postings_purged
        """).fetchone()[0]

        print(
            f"Postings after bucket purge: "
            f"{purged_postings:,}"
        )

        # ====================================================
        # REFERENCE INDEX
        #
        # Keep TOP-K rarest surviving address tokens
        # per reference entity.
        #
        # QUALIFY avoids the previous nested-alias issue.
        # ====================================================

        con.execute("DROP TABLE IF EXISTS ref_index")

        con.execute(f"""
            CREATE TEMP TABLE ref_index AS

            SELECT
                p.entity_id,
                p.country,
                p.token

            FROM postings_purged AS p

            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY p.entity_id
                ORDER BY p.token_df, p.token
            ) <= {top_k}
        """)

        ref_index_rows = con.execute("""
            SELECT COUNT(*)
            FROM ref_index
        """).fetchone()[0]

        print(
            f"Reference index rows: "
            f"{ref_index_rows:,}"
        )

        # ====================================================
        # S1 ADDRESS TOKENS
        # ====================================================

        con.execute("DROP TABLE IF EXISTS s1_tokens")

        con.execute("""
            CREATE TEMP TABLE s1_tokens AS

            SELECT DISTINCT
                s.entity_id,
                s.country,
                LOWER(TRIM(t.token)) AS token

            FROM pilot_s1 AS s

            CROSS JOIN UNNEST(
                regexp_split_to_array(
                    COALESCE(s.address, ''),
                    '[^[:alnum:]]+'
                )
            ) AS t(token)

            WHERE
                TRIM(t.token) <> ''
                AND LENGTH(TRIM(t.token)) >= 2
        """)

        # ====================================================
        # S1 INDEX
        #
        # Keep TOP-K rarest surviving address tokens
        # per S1 entity.
        # ====================================================

        con.execute("DROP TABLE IF EXISTS s1_index")

        con.execute(f"""
            CREATE TEMP TABLE s1_index AS

            SELECT
                s.entity_id,
                s.country,
                s.token

            FROM s1_tokens AS s

            INNER JOIN good_tokens AS g
                ON s.token = g.token

            INNER JOIN good_bucket_tokens AS b
                ON s.token = b.token

            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY s.entity_id
                ORDER BY g.token_df, s.token
            ) <= {top_k}
        """)

        # ====================================================
        # CANDIDATE GENERATION
        #
        # Same country + shared surviving address token.
        # ====================================================

        con.execute("DROP TABLE IF EXISTS candidates")

        con.execute("""
            CREATE TEMP TABLE candidates AS

            SELECT DISTINCT
                s.entity_id AS source1_entity_id,
                r.entity_id AS candidate_entity_id

            FROM s1_index AS s

            INNER JOIN ref_index AS r
                ON s.country = r.country
               AND s.token = r.token
        """)

        candidate_pairs = con.execute("""
            SELECT COUNT(*)
            FROM candidates
        """).fetchone()[0]

        s1_with_candidates = con.execute("""
            SELECT COUNT(DISTINCT source1_entity_id)
            FROM candidates
        """).fetchone()[0]

        coverage = (
            s1_with_candidates / pilot_rows * 100.0
            if pilot_rows
            else 0.0
        )

        avg_candidates = (
            candidate_pairs / pilot_rows
            if pilot_rows
            else 0.0
        )

        # ====================================================
        # CANDIDATE DISTRIBUTION
        # ====================================================

        distribution = con.execute("""
            SELECT
                quantile_cont(cnt, 0.50),
                quantile_cont(cnt, 0.90),
                quantile_cont(cnt, 0.95),
                quantile_cont(cnt, 0.99),
                quantile_cont(cnt, 0.999),
                MAX(cnt)

            FROM (
                SELECT
                    source1_entity_id,
                    COUNT(*) AS cnt

                FROM candidates

                GROUP BY source1_entity_id
            )
        """).fetchone()

        p50 = distribution[0] or 0
        p90 = distribution[1] or 0
        p95 = distribution[2] or 0
        p99 = distribution[3] or 0
        p999 = distribution[4] or 0
        max_candidates = distribution[5] or 0

        # ====================================================
        # BLOCKING RECALL
        # ====================================================

        recovered = con.execute("""
            SELECT COUNT(*)

            FROM gt AS g

            INNER JOIN pilot_s1 AS s
                ON g.source1_entity_id = s.entity_id

            INNER JOIN candidates AS c
                ON c.source1_entity_id = g.source1_entity_id
               AND c.candidate_entity_id = g.candidate_entity_id
        """).fetchone()[0]

        blocking_recall = (
            recovered / pilot_true_pairs * 100.0
            if pilot_true_pairs
            else 0.0
        )

        # ====================================================
        # RESULTS
        # ====================================================

        elapsed = mins(start_cfg)

        print()
        print(f"S1 with candidates: {s1_with_candidates:,}")
        print(f"Coverage:            {coverage:.4f}%")
        print(f"Candidate pairs:     {candidate_pairs:,}")
        print(f"Avg candidates/S1:   {avg_candidates:.2f}")
        print()
        print("Candidate distribution:")
        print(f"  P50:   {p50:.0f}")
        print(f"  P90:   {p90:.0f}")
        print(f"  P95:   {p95:.0f}")
        print(f"  P99:   {p99:.0f}")
        print(f"  P99.9: {p999:.0f}")
        print(f"  MAX:   {max_candidates:,.0f}")
        print()
        print(f"True pairs:          {pilot_true_pairs:,}")
        print(f"Recovered true:      {recovered:,}")
        print(f"Blocking recall:     {blocking_recall:.4f}%")
        print(f"Runtime:              {elapsed:.1f} min")
        print()

        results.append({
            "config": name,
            "df_cap": df_cap,
            "top_k": top_k,
            "bucket_cap": bucket_cap,
            "coverage": coverage,
            "candidate_pairs": candidate_pairs,
            "avg_candidates": avg_candidates,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "p999": p999,
            "max_candidates": max_candidates,
            "recovered": recovered,
            "blocking_recall": blocking_recall,
            "runtime_min": elapsed,
        })

        # ====================================================
        # CLEAN CONFIG-SPECIFIC TABLES
        # ====================================================

        # Keep Config A candidates for union analysis.
        if name != "A":
            con.execute("DROP TABLE IF EXISTS candidates")
        con.execute("DROP TABLE IF EXISTS s1_index")
        con.execute("DROP TABLE IF EXISTS s1_tokens")
        con.execute("DROP TABLE IF EXISTS ref_index")
        con.execute("DROP TABLE IF EXISTS postings_purged")
        con.execute("DROP TABLE IF EXISTS good_bucket_tokens")
        con.execute("DROP TABLE IF EXISTS bucket_sizes")
        con.execute("DROP TABLE IF EXISTS postings")


    # ========================================================
    # V2 + A TRUE-PAIR UNION ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("V2 + A UNION ANALYSIS")
    print("=" * 70)

    # Load V2 candidates.
    con.execute("DROP TABLE IF EXISTS v2_analysis")

    con.execute("""
        CREATE TEMP TABLE v2_analysis AS

        SELECT DISTINCT
            TRIM(source1_entity_id) AS source1_entity_id,
            TRIM(candidate_entity_id) AS candidate_entity_id

        FROM (
            SELECT
                source1_entity_id,
                UNNEST(
                    regexp_split_to_array(
                        COALESCE(candidate_entity_ids, ''),
                        '[,;]'
                    )
                ) AS candidate_entity_id

            FROM read_csv(
                'output/candidate_pairs_validation_top2.tsv',
                delim='\t',
                header=true,
                all_varchar=true,
                ignore_errors=true
            )
        )

        WHERE TRIM(candidate_entity_id) <> ''
    """)

    v2_pairs = con.execute("""
        SELECT COUNT(*)
        FROM v2_analysis
    """).fetchone()[0]

    pilot_true = con.execute("""
        SELECT COUNT(*)
        FROM gt g
        INNER JOIN pilot_s1 s
            ON g.source1_entity_id = s.entity_id
    """).fetchone()[0]

    # V2 true-pair recall.
    v2_recovered = con.execute("""
        SELECT COUNT(*)

        FROM gt g

        INNER JOIN v2_analysis v
            ON g.source1_entity_id = v.source1_entity_id
           AND g.candidate_entity_id = v.candidate_entity_id

        INNER JOIN pilot_s1 s
            ON g.source1_entity_id = s.entity_id
    """).fetchone()[0]

    # A true-pair recall.
    a_recovered = con.execute("""
        SELECT COUNT(*)

        FROM gt g

        INNER JOIN address_a a
            ON g.source1_entity_id = a.source1_entity_id
           AND g.candidate_entity_id = a.candidate_entity_id

        INNER JOIN pilot_s1 s
            ON g.source1_entity_id = s.entity_id
    """).fetchone()[0]

    # True pairs A recovers that V2 misses.
    a_recovers_v2_misses = con.execute("""
        SELECT COUNT(*)

        FROM gt g

        INNER JOIN pilot_s1 s
            ON g.source1_entity_id = s.entity_id

        INNER JOIN address_a a
            ON g.source1_entity_id = a.source1_entity_id
           AND g.candidate_entity_id = a.candidate_entity_id

        WHERE NOT EXISTS (
            SELECT 1

            FROM v2_analysis v

            WHERE
                v.source1_entity_id = g.source1_entity_id
                AND v.candidate_entity_id = g.candidate_entity_id
        )
    """).fetchone()[0]

    union_recovered = v2_recovered + a_recovers_v2_misses

    v2_recall = v2_recovered / pilot_true * 100
    a_recall = a_recovered / pilot_true * 100
    union_recall = union_recovered / pilot_true * 100

    print()
    print(f"Pilot true pairs:             {pilot_true:,}")
    print()
    print(
        f"V2 recovered:                 "
        f"{v2_recovered:,} ({v2_recall:.4f}%)"
    )
    print(
        f"A recovered:                  "
        f"{a_recovered:,} ({a_recall:.4f}%)"
    )
    print()
    print(
        f"A recovers V2 misses:         "
        f"{a_recovers_v2_misses:,}"
    )
    print()
    print(
        f"V2 + A union recovered:       "
        f"{union_recovered:,} ({union_recall:.4f}%)"
    )
    print(
        f"Recall gain over V2:           "
        f"{union_recall - v2_recall:+.4f} pp"
    )
    print()
    print(f"V2 candidates:                 {v2_pairs:,}")
    print(f"A candidates:                  {candidate_pairs:,}")
    print(
        f"Naive V2+A candidate total:    "
        f"{v2_pairs + candidate_pairs:,}"
    )
    print("=" * 70)

    con.execute("DROP TABLE IF EXISTS v2_analysis")
    con.execute("DROP TABLE IF EXISTS address_a")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"{'CFG':<5}"
        f"{'Recall':>12}"
        f"{'Coverage':>12}"
        f"{'Pairs':>16}"
        f"{'Avg/S1':>12}"
        f"{'Runtime':>10}"
    )

    print("-" * 70)

    for r in results:
        print(
            f"{r['config']:<5}"
            f"{r['blocking_recall']:>11.4f}%"
            f"{r['coverage']:>11.4f}%"
            f"{r['candidate_pairs']:>16,}"
            f"{r['avg_candidates']:>12.2f}"
            f"{r['runtime_min']:>9.1f}m"
        )

    print("=" * 70)

    con.close()


if __name__ == "__main__":
    main()