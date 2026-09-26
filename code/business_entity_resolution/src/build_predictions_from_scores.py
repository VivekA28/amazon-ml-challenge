import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--threshold", required=True, type=float)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()

    first = True
    total = 0
    kept = 0

    for chunk in pd.read_csv(
        args.scores,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
        },
        chunksize=args.chunksize,
    ):
        total += len(chunk)

        chunk = chunk[chunk["score"] >= args.threshold]

        if len(chunk) == 0:
            continue

        # matching_results format:
        # source1_entity_id    candidate_entity_ids
        grouped = (
            chunk.groupby("source1_entity_id")["candidate_entity_id"]
            .agg(lambda x: ",".join(dict.fromkeys(x)))
            .reset_index()
        )

        grouped.columns = [
            "source1_entity_id",
            "candidate_entity_ids",
        ]

        grouped.to_csv(
            args.output,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )

        first = False
        kept += len(chunk)

        print(
            f"Processed {total:,} rows | "
            f"kept {kept:,} pairs",
            flush=True,
        )

    print(f"Total scored pairs: {total:,}")
    print(f"Pairs >= threshold: {kept:,}")
    print(f"Predictions: {args.output}")


if __name__ == "__main__":
    main()