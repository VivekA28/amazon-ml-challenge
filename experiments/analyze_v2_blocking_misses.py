from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


NON_ALNUM = re.compile(r"[^a-z0-9]+")
NUMBER = re.compile(r"\b\d+[a-z]?\b")


def norm(x):
    x = "" if x is None else str(x)
    x = x.lower().strip().replace("&", "and")
    x = NON_ALNUM.sub(" ", x)
    return " ".join(x.split())


def tokens(x):
    return set(norm(x).split())


def ratio(a, b):
    # difflib is enough for this diagnostic
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def first_number(x):
    m = NUMBER.search(norm(x))
    return m.group(0) if m else ""


def load_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def parse_gt(value):
    if not value:
        return []
    return [x for x in str(value).split(",") if x]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--s1", required=True, type=Path)
    ap.add_argument("--source2", required=True, type=Path)
    ap.add_argument("--source3", required=True, type=Path)
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--candidates", required=True, type=Path)

    args = ap.parse_args()

    print("[1/5] Loading validation S1...")
    s1 = load_tsv(args.s1)

    print("[2/5] Loading S2 + S3...")
    s2 = load_tsv(args.source2)
    s3 = load_tsv(args.source3)

    ref = pd.concat([s2, s3], ignore_index=True)

    # ------------------------------------------------------------
    # Build lookup
    # ------------------------------------------------------------

    ref = ref[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    ref["name_norm"] = ref["business_name"].map(norm)
    ref["addr_norm"] = ref["business_address"].map(norm)
    ref["country_norm"] = ref["country"].map(norm)

    ref_by_id = ref.set_index("entity_id")

    print(f"Reference records: {len(ref):,}")

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    print("[3/5] Loading ground truth...")

    gt = load_tsv(args.ground_truth)

    gt_map = {}

    for _, row in gt.iterrows():
        gt_map[row["source1_entity_id"]] = parse_gt(
            row["matched_entity_ids"]
        )

    # ------------------------------------------------------------
    # Candidate set
    # ------------------------------------------------------------

    print("[4/5] Loading V2 candidates...")

    cand = load_tsv(args.candidates)

    # Candidate file should have source1_entity_id/candidate_entity_id
    cand_map = {}

    for _, row in cand.iterrows():
     cand_map[row["source1_entity_id"]] = set(
        parse_gt(row["candidate_entity_ids"])
    )

    # ------------------------------------------------------------
    # Analyze TRUE pairs missing from V2 candidates
    # ------------------------------------------------------------

    print("[5/5] Analyzing missing true pairs...")

    rows = []

    total_true = 0
    missing_true = 0

    for _, src in s1.iterrows():

        sid = src["entity_id"]

        true_ids = gt_map.get(sid, [])
        candidates = cand_map.get(sid, set())

        for tid in true_ids:

            total_true += 1

            if tid in candidates:
                continue

            if tid not in ref_by_id.index:
                continue

            target = ref_by_id.loc[tid]

            s1_name = norm(src["business_name"])
            s2_name = target["name_norm"]

            s1_addr = norm(src["business_address"])
            s2_addr = target["addr_norm"]

            s1_name_tok = tokens(s1_name)
            s2_name_tok = tokens(s2_name)

            s1_addr_tok = tokens(s1_addr)
            s2_addr_tok = tokens(s2_addr)

            name_exact = int(s1_name == s2_name)

            name_j = jaccard(
                s1_name_tok,
                s2_name_tok,
            )

            name_o = overlap(
                s1_name_tok,
                s2_name_tok,
            )

            name_r = ratio(
                s1_name,
                s2_name,
            )

            addr_exact = int(s1_addr == s2_addr)

            addr_j = jaccard(
                s1_addr_tok,
                s2_addr_tok,
            )

            addr_o = overlap(
                s1_addr_tok,
                s2_addr_tok,
            )

            addr_r = ratio(
                s1_addr,
                s2_addr,
            )

            country_equal = int(
                norm(src["country"])
                == target["country_norm"]
            )

            n1 = first_number(s1_addr)
            n2 = first_number(s2_addr)

            number_equal = int(
                bool(n1)
                and bool(n2)
                and n1 == n2
            )

            shared_name = s1_name_tok & s2_name_tok
            shared_addr = s1_addr_tok & s2_addr_tok

            rows.append(
                {
                    "source1_entity_id": sid,
                    "candidate_entity_id": tid,

                    "name_exact": name_exact,
                    "name_jaccard": name_j,
                    "name_overlap": name_o,
                    "name_ratio": name_r,

                    "address_exact": addr_exact,
                    "address_jaccard": addr_j,
                    "address_overlap": addr_o,
                    "address_ratio": addr_r,

                    "country_equal": country_equal,
                    "address_number_equal": number_equal,

                    "shared_name_tokens": len(shared_name),
                    "shared_address_tokens": len(shared_addr),

                    "s1_name": s1_name,
                    "true_name": s2_name,
                    "s1_address": s1_addr,
                    "true_address": s2_addr,
                }
            )

            missing_true += 1

    out = pd.DataFrame(rows)

    out_path = Path(
        "experiments/v2_missing_true_pairs.tsv"
    )

    out.to_csv(
        out_path,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 78)
    print("V2 BLOCKING MISS ANALYSIS")
    print("=" * 78)

    print(f"Total true pairs       : {total_true:,}")
    print(f"Missing from candidates: {missing_true:,}")

    if total_true:
        print(
            f"Missing rate           : "
            f"{100 * missing_true / total_true:.4f}%"
        )

    if out.empty:
        print("No missing true pairs.")
        return

    print()
    print("MISSING-PAIR CHARACTERISTICS")
    print("-" * 78)

    checks = {
        "name_exact": out["name_exact"] == 1,

        "name_ratio >= .95":
            out["name_ratio"] >= 0.95,

        "name_ratio >= .90":
            out["name_ratio"] >= 0.90,

        "name_ratio >= .80":
            out["name_ratio"] >= 0.80,

        "name_jaccard >= .80":
            out["name_jaccard"] >= 0.80,

        "name_jaccard >= .60":
            out["name_jaccard"] >= 0.60,

        "shared_name_token":
            out["shared_name_tokens"] >= 1,

        "address_number_equal":
            out["address_number_equal"] == 1,

        "address_ratio >= .90":
            out["address_ratio"] >= 0.90,

        "address_ratio >= .80":
            out["address_ratio"] >= 0.80,

        "address_jaccard >= .60":
            out["address_jaccard"] >= 0.60,

        "shared_address_token":
            out["shared_address_tokens"] >= 1,

        "country_equal":
            out["country_equal"] == 1,
    }

    for label, mask in checks.items():

        count = int(mask.sum())

        print(
            f"{label:<28}: "
            f"{count:>8,} "
            f"({100 * count / len(out):6.2f}%)"
        )

    print()
    print("NAME SIMILARITY DISTRIBUTION")
    print("-" * 78)

    print(
        out["name_ratio"].describe(
            percentiles=[
                .01,
                .10,
                .25,
                .50,
                .75,
                .90,
                .95,
                .99,
            ]
        )
    )

    print()
    print("ADDRESS SIMILARITY DISTRIBUTION")
    print("-" * 78)

    print(
        out["address_ratio"].describe(
            percentiles=[
                .01,
                .10,
                .25,
                .50,
                .75,
                .90,
                .95,
                .99,
            ]
        )
    )

    print()
    print("HIGH-VALUE COMBINATIONS")
    print("-" * 78)

    combos = {
        "name_ratio>=.90 AND country":
            (out["name_ratio"] >= .90)
            & (out["country_equal"] == 1),

        "name_ratio>=.80 AND country":
            (out["name_ratio"] >= .80)
            & (out["country_equal"] == 1),

        "name_jaccard>=.60 AND country":
            (out["name_jaccard"] >= .60)
            & (out["country_equal"] == 1),

        "name_ratio>=.80 AND number":
            (out["name_ratio"] >= .80)
            & (out["address_number_equal"] == 1),

        "name_ratio>=.80 AND shared address":
            (out["name_ratio"] >= .80)
            & (out["shared_address_tokens"] >= 1),

        "name_ratio>=.90 AND number":
            (out["name_ratio"] >= .90)
            & (out["address_number_equal"] == 1),
    }

    for label, mask in combos.items():

        count = int(mask.sum())

        print(
            f"{label:<38}: "
            f"{count:>8,} "
            f"({100 * count / len(out):6.2f}%)"
        )

    print()
    print(f"Saved: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
