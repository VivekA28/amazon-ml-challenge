from pathlib import Path
from collections import defaultdict
import re
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = 100_000
PILOT_S1 = 100_000
MIN_TOKEN_LEN = 4
MAX_POSTING = 50_000

_non_alnum = re.compile(r"[^a-z0-9]+")
_number = re.compile(r"\b\d+[a-z]?\b")


def norm(value):
    value = str(value).lower().strip().replace("&", "and")
    value = _non_alnum.sub(" ", value)
    return " ".join(value.split())


def tokens(value):
    return {x for x in norm(value).split() if len(x) >= MIN_TOKEN_LEN}


def addr_number(value):
    m = _number.search(norm(value))
    return m.group(0) if m else None


def load_pilot():
    s1_path = DATA_DIR / "train" / "train_source1.tsv"
    gt_path = DATA_DIR / "train" / "train_ground_truth.tsv"

    s1 = next(pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        chunksize=PILOT_S1,
        keep_default_na=False,
    )).iloc[:PILOT_S1].copy()

    ids = set(s1.entity_id)
    truth = {}

    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        sub = chunk[chunk.source1_entity_id.isin(ids)]
        for r in sub.itertuples(index=False):
            truth[r.source1_entity_id] = (
                set(r.matched_entity_ids.split(","))
                if r.matched_entity_ids else set()
            )
        if len(truth) == len(ids):
            break

    return s1, truth


def build_indexes():
    sources = [
        DATA_DIR / "train" / "train_source2.tsv",
        DATA_DIR / "train" / "train_source3.tsv",
    ]

    exact_name = defaultdict(set)
    exact_addr = defaultdict(set)
    country_number = defaultdict(set)
    country_name = defaultdict(set)
    country_addr = defaultdict(set)

    print("=" * 78)
    print("EXP-006 LITE: BUILDING ADDRESS INDEXES")
    print("=" * 78)

    overall = time.time()

    for path in sources:
        started = time.time()
        print(f"Indexing {path.name} ...", flush=True)

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):
            # Normalize once per chunk instead of repeatedly inside index loops.
            names = chunk.business_name.map(norm)
            addrs = chunk.business_address.map(norm)

            for eid, country, name, addr in zip(
                chunk.entity_id,
                chunk.country,
                names,
                addrs,
            ):
                if name:
                    exact_name[name].add(eid)

                    for tok in tokens(name):
                        country_name[(country, tok)].add(eid)

                if addr:
                    exact_addr[addr].add(eid)

                    num = addr_number(addr)
                    if num:
                        country_number[(country, num)].add(eid)

                    for tok in tokens(addr):
                        country_addr[(country, tok)].add(eid)

        print(
            f"  done in {(time.time()-started)/60:.1f} min",
            flush=True,
        )

    print(
        f"All indexes built in {(time.time()-overall)/60:.1f} min",
        flush=True,
    )

    return exact_name, exact_addr, country_number, country_name, country_addr


def evaluate(
    s1,
    truth,
    exact_name,
    exact_addr,
    country_number,
    country_name,
    country_addr,
):
    names = [
        "A_exact_address",
        "B_number_name",
        "C_address_name",
        "D_name_plus_B",
        "E_name_plus_C",
    ]

    stats = {
        k: {"candidates": 0, "covered": 0, "recovered": 0}
        for k in names
    }

    total_true = sum(len(truth.get(eid, set())) for eid in s1.entity_id)

    started = time.time()

    for i, r in enumerate(s1.itertuples(index=False), 1):
        country = r.country
        name = norm(r.business_name)
        addr = norm(r.business_address)

        # Candidate name tokens, selecting the smallest posting list.
        name_postings = []
        for tok in tokens(name):
            posting = country_name.get((country, tok), set())
            if posting and len(posting) <= MAX_POSTING:
                name_postings.append((len(posting), posting))

        name_postings.sort(key=lambda x: x[0])
        name_set = name_postings[0][1] if name_postings else set()

        # A: exact address.
        a = exact_addr.get(addr, set()) if addr else set()

        # B: country + address number AND rare-ish name token.
        num = addr_number(addr)
        num_set = (
            country_number.get((country, num), set())
            if num else set()
        )
        b = num_set & name_set

        # C: country + smallest address posting AND name posting.
        c = set()
        if name_set:
            addr_postings = []
            for tok in tokens(addr):
                posting = country_addr.get((country, tok), set())
                if posting and len(posting) <= MAX_POSTING:
                    addr_postings.append((len(posting), posting))

            if addr_postings:
                _, addr_set = min(addr_postings, key=lambda x: x[0])
                c = addr_set & name_set

        exact_n = exact_name.get(name, set()) if name else set()

        variants = {
            "A_exact_address": a,
            "B_number_name": b,
            "C_address_name": c,
            "D_name_plus_B": exact_n | b,
            "E_name_plus_C": exact_n | c,
        }

        true = truth.get(r.entity_id, set())

        for key, cand in variants.items():
            stats[key]["candidates"] += len(cand)
            stats[key]["covered"] += bool(cand)
            stats[key]["recovered"] += len(cand & true)

        if i % 25_000 == 0:
            print(f"Evaluated {i:,}/{len(s1):,}", flush=True)

    elapsed = time.time() - started

    print("\n" + "=" * 92)
    print(f"EXP-006 LITE RESULTS — {len(s1):,} S1 PILOT")
    print("=" * 92)
    print(
        f"{'Strategy':<24} {'Coverage':>10} "
        f"{'Avg cand/S1':>14} {'Recovered':>12} {'Recall':>12}"
    )
    print("-" * 92)

    for key in names:
        x = stats[key]
        coverage = x["covered"] / len(s1)
        avg = x["candidates"] / len(s1)
        recall = x["recovered"] / total_true if total_true else 0

        print(
            f"{key:<24} {coverage:>9.2%} "
            f"{avg:>14,.2f} {x['recovered']:>12,} "
            f"{recall:>11.2%}"
        )

    print("-" * 92)
    print(f"True matches in pilot: {total_true:,}")
    print(f"Evaluation time: {elapsed:.1f}s")
    print("=" * 92)


if __name__ == "__main__":
    overall = time.time()

    s1, truth = load_pilot()

    indexes = build_indexes()

    evaluate(
        s1,
        truth,
        *indexes,
    )

    print(
        f"\nTOTAL RUNTIME: {(time.time()-overall)/60:.1f} min"
    )
