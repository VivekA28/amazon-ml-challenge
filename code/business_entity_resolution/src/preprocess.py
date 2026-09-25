"""
Shared preprocessing / normalization pipeline.

This is the ONE canonical normalization layer for the project. Every
blocking experiment and model should read the *_normalized.parquet
output of this script rather than re-implementing normalize_name /
normalize_address / normalize_country locally.

Design goals:
- Streaming / chunked: never loads the full 24.4M-row dataset into memory.
- Deterministic: same input row -> same normalized output, always.
- Non-destructive: original business_name / business_address / country
  columns are preserved alongside normalized versions, so any experiment
  that wants raw text still has it.
- Country-agnostic: no hard-coded {US, India} set. Test data adds France,
  and the challenge could add more; normalization must not assume a
  fixed country list.

Output schema (all string columns except where noted):
    entity_id
    business_name              (original, unmodified)
    business_name_norm         (lowercased, unicode-normalized, cleaned)
    business_name_core         (business_name_norm with legal suffix stripped)
    legal_suffix                (detected legal suffix token, "" if none)
    business_address           (original, unmodified)
    business_address_norm      (lowercased, unicode-normalized, abbrevs expanded)
    country                    (original, unmodified)
    country_norm                (canonicalized country code/label)

Usage:
    python3 preprocess.py \
        --input data/train/train_source2.tsv \
        --output processed/train_source2.parquet

    python3 preprocess.py \
        --input data/test/test_source1.tsv \
        --output processed/test_source1.parquet \
        --chunk-size 200000

Run once per source file (S1/S2/S3, train/test = 6 runs total). Output is
Parquet (columnar, compressed, fast to re-read) instead of TSV so repeated
downstream experiments don't pay re-parsing cost on 24M rows every time.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import re
import time
import unicodedata

import pandas as pd

# pyarrow is only needed for the actual Parquet-writing step, imported
# lazily inside preprocess_file(). This keeps normalize_name/_address/
# _country and process_chunk importable and unit-testable even in an
# environment where pyarrow isn't installed yet.


# ============================================================
# Configuration
# ============================================================

DEFAULT_CHUNK_SIZE = 100_000

# Legal suffixes are extracted as a SEPARATE feature, not deleted blindly.
# Ordered longest-first (as tokens) so multi-word suffixes match before
# their single-word substrings (e.g. "pvt ltd" before "ltd").
LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt ltd.",
    "public limited company", "plc",
    "limited liability company", "limited liability partnership",
    "llp", "llc", "l.l.c.",
    "incorporated", "inc.", "inc",
    "corporation", "corp.", "corp",
    "company", "co.", "co",
    "limited", "ltd.", "ltd",
    "gmbh", "ag", "kg", "sarl", "srl", "spa", "s.p.a.",
    "bv", "b.v.", "nv", "n.v.",
    "sa", "s.a.",
    "pty ltd", "pty. ltd.", "pty",
]
# Pre-tokenized, longest-first, for greedy suffix matching against the
# END of the normalized name.
_LEGAL_SUFFIX_TOKENS = sorted(
    (re.sub(r"[^\w\s]", "", s).split() for s in LEGAL_SUFFIXES),
    key=len,
    reverse=True,
)

# Common address abbreviations. Applied as whole-token replacements only
# (never substring), so "drive" inside a longer word is never touched.
ADDRESS_ABBREVIATIONS = {
    "st": "street", "str": "street",
    "ave": "avenue", "av": "avenue",
    "rd": "road",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "hwy": "highway",
    "pkwy": "parkway",
    "ter": "terrace",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mt": "mount",
    "ft": "fort",
    "jn": "junction",
    "nr": "near",
    "opp": "opposite",
}

# Minimal set of well-known country aliases -> canonical label. This is
# intentionally small and NOT exhaustive by design: normalize_country()
# falls back to a generic cleanup (unicode-normalize, trim, collapse
# whitespace, uppercase 2-letter codes) for anything not listed, so a
# country appearing only in test (e.g. France) or any future country is
# still handled sanely without a code change.
COUNTRY_ALIASES = {
    "us": "US", "u.s.": "US", "u.s.a.": "US", "usa": "US",
    "united states": "US", "united states of america": "US",
    "in": "IN", "india": "IN", "bharat": "IN",
    "fr": "FR", "france": "FR",
    "uk": "GB", "u.k.": "GB", "united kingdom": "GB", "great britain": "GB",
    "de": "DE", "germany": "DE", "deutschland": "DE",
}


# ============================================================
# Text normalization primitives
# ============================================================

def _unicode_clean(text: str) -> str:
    """NFKC unicode normalization, so visually-identical characters from
    different encodings collapse to one representation."""
    return unicodedata.normalize("NFKC", text)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(raw_name: str) -> tuple[str, str, str]:
    """
    Returns (business_name_norm, business_name_core, legal_suffix).

    business_name_norm : cleaned but suffix-preserving normalized name.
    business_name_core : business_name_norm with a trailing legal suffix
                          removed, for blocking/matching that should
                          ignore entity-type differences.
    legal_suffix        : the detected suffix (e.g. "llc"), "" if none.
    """
    if not raw_name:
        return "", "", ""

    text = _unicode_clean(raw_name).lower().strip()
    text = text.replace("&", " and ")

    # Punctuation -> space (keep alphanumerics; this also normalizes
    # commas/periods/hyphens consistently instead of deleting them and
    # accidentally gluing two words together).
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = _collapse_whitespace(text)

    tokens = text.split()
    legal_suffix = ""

    if tokens:
        for suffix_tokens in _LEGAL_SUFFIX_TOKENS:
            n = len(suffix_tokens)
            if n and tokens[-n:] == suffix_tokens:
                legal_suffix = " ".join(suffix_tokens)
                break

    if legal_suffix:
        core_tokens = tokens[: len(tokens) - len(legal_suffix.split())]
        core = " ".join(core_tokens).strip()
    else:
        core = text

    return text, core, legal_suffix


def normalize_address(raw_address: str) -> str:
    """Lowercase, unicode-clean, expand common abbreviations, preserve
    numbers exactly, normalize whitespace/punctuation without deleting
    informative characters (e.g. keep house/unit numbers intact)."""
    if not raw_address:
        return ""

    text = _unicode_clean(raw_address).lower().strip()

    # Keep alphanumerics, spaces, and '#' (common unit-number marker);
    # turn everything else into a space so "12,Main St." doesn't merge
    # into "12main st".
    text = re.sub(r"[^\w\s#]", " ", text, flags=re.UNICODE)
    text = _collapse_whitespace(text)

    tokens = [
        ADDRESS_ABBREVIATIONS.get(tok, tok)
        for tok in text.split()
    ]
    return " ".join(tokens)


def normalize_country(raw_country: str) -> str:
    """Canonicalize a country field WITHOUT assuming a fixed country set.
    Known aliases map to a short canonical code; anything unrecognized is
    still cleaned (unicode, case, whitespace) and passed through so new
    countries (e.g. France in test) degrade gracefully instead of being
    dropped or mis-bucketed."""
    if not raw_country:
        return ""

    text = _unicode_clean(raw_country).strip().lower()
    text = _collapse_whitespace(text)

    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]

    # Already a bare 2-letter code (e.g. "fr") that isn't in the alias
    # table yet -> uppercase it rather than leaving it lowercase.
    if len(text) == 2 and text.isalpha():
        return text.upper()

    # Unknown longer label: title-case it for consistency, don't guess.
    return text.title()


# ============================================================
# Streaming chunk processing
# ============================================================

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    names = chunk["business_name"].fillna("")
    addresses = chunk["business_address"].fillna("")
    countries = chunk["country"].fillna("")

    name_results = names.map(normalize_name)
    business_name_norm = name_results.map(lambda t: t[0])
    business_name_core = name_results.map(lambda t: t[1])
    legal_suffix = name_results.map(lambda t: t[2])

    business_address_norm = addresses.map(normalize_address)
    country_norm = countries.map(normalize_country)

    return pd.DataFrame(
        {
            "entity_id": chunk["entity_id"],
            "business_name": chunk["business_name"],
            "business_name_norm": business_name_norm,
            "business_name_core": business_name_core,
            "legal_suffix": legal_suffix,
            "business_address": chunk["business_address"],
            "business_address_norm": business_address_norm,
            "country": chunk["country"],
            "country_norm": country_norm,
        }
    )


def preprocess_file(
    input_path: Path,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: "pq.ParquetWriter | None" = None
    total_rows = 0
    start_time = time.time()

    try:
        for chunk in pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            chunksize=chunk_size,
            keep_default_na=False,
        ):
            processed = process_chunk(chunk)
            table = pa.Table.from_pandas(processed, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path, table.schema, compression="snappy"
                )

            writer.write_table(table)

            total_rows += len(processed)
            elapsed = time.time() - start_time
            print(
                f"  {input_path.name}: {total_rows:,} rows "
                f"({elapsed / 60:.1f} min)"
            )
    finally:
        if writer is not None:
            writer.close()

    print(
        f"Done: {input_path} -> {output_path} "
        f"({total_rows:,} rows, {(time.time() - start_time) / 60:.1f} min)"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    print(f"Preprocessing {args.input} -> {args.output}")
    preprocess_file(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
