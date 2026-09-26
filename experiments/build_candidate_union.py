#!/usr/bin/env python3

import csv
from pathlib import Path

BASE = Path("/home/vivek/Desktop/amazon-ml-challenge")

V2 = BASE / "output/candidate_pairs_validation_top2.tsv"
F2 = BASE / "output/candidate_pairs_validation_f2.tsv"
OUT = BASE / "output/candidate_pairs_validation_union.tsv"


def load_v2():
    data = {}

    with V2.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader, 1):
            sid = row["source1_entity_id"]
            ids = row["candidate_entity_ids"]

            data[sid] = set(x for x in ids.split(",") if x)

            if i % 50000 == 0:
                print(f"[load V2] S1 {i:,}", flush=True)

    return data


def main():
    print("[1/3] Loading V2 candidates...")
    candidates = load_v2()

    print(f"[1/3] Loaded {len(candidates):,} S1 entities")

    total_union_pairs = 0
    max_candidates = 0
    f2_rows = 0

    print("[2/3] Streaming F2 and building union...")

    with F2.open("r", encoding="utf-8", newline="") as fin, \
         OUT.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_ids"
        ])

        for row in reader:
            sid = row["source1_entity_id"]

            if sid not in candidates:
                raise RuntimeError(
                    f"F2 contains unknown S1: {sid}"
                )

            ids = row["candidate_entity_ids"]

            candidates[sid].update(
                x for x in ids.split(",") if x
            )

            f2_rows += 1

            if f2_rows % 50000 == 0:
                print(
                    f"[merge] F2 S1 {f2_rows:,}",
                    flush=True
                )

        print("[3/3] Writing union...")

        for i, (sid, ids) in enumerate(candidates.items(), 1):
            ids = sorted(ids)

            writer.writerow([
                sid,
                ",".join(ids)
            ])

            n = len(ids)
            total_union_pairs += n
            max_candidates = max(max_candidates, n)

            if i % 50000 == 0:
                print(
                    f"[write] S1 {i:,} | "
                    f"union pairs {total_union_pairs:,}",
                    flush=True
                )

    print("\n" + "=" * 72)
    print("CANDIDATE UNION COMPLETE")
    print("=" * 72)
    print(f"S1 rows:          {len(candidates):,}")
    print(f"F2 rows processed: {f2_rows:,}")
    print(f"Union pairs:      {total_union_pairs:,}")
    print(f"Max candidates/S1: {max_candidates:,}")
    print(f"Output:           {OUT}")


if __name__ == "__main__":
    main()
