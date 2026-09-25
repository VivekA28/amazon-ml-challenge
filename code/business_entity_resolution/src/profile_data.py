from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = 100_000


def profile_source(path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"FILE: {path.name}")
    print(f"{'=' * 70}")

    total_rows = 0
    missing_name = 0
    missing_address = 0
    missing_country = 0

    countries = Counter()

    name_chars = 0
    address_chars = 0

    name_min = float("inf")
    name_max = 0
    address_min = float("inf")
    address_max = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        total_rows += len(chunk)

        names = chunk["business_name"].str.strip()
        addresses = chunk["business_address"].str.strip()
        country = chunk["country"].str.strip()

        missing_name += (names == "").sum()
        missing_address += (addresses == "").sum()
        missing_country += (country == "").sum()

        countries.update(country)

        name_lengths = names.str.len()
        address_lengths = addresses.str.len()

        name_chars += name_lengths.sum()
        address_chars += address_lengths.sum()

        name_min = min(name_min, name_lengths.min())
        name_max = max(name_max, name_lengths.max())

        address_min = min(address_min, address_lengths.min())
        address_max = max(address_max, address_lengths.max())

    print(f"Rows:              {total_rows:,}")
    print(f"Missing names:     {missing_name:,}")
    print(f"Missing addresses: {missing_address:,}")
    print(f"Missing countries: {missing_country:,}")

    print("\nCountry distribution:")
    for country, count in countries.most_common():
        label = country if country else "[EMPTY]"
        print(f"  {label:15} {count:,}")

    print("\nName length:")
    print(f"  Min:    {name_min:,}")
    print(f"  Max:    {name_max:,}")
    print(f"  Mean:   {name_chars / total_rows:.2f}")

    print("\nAddress length:")
    print(f"  Min:    {address_min:,}")
    print(f"  Max:    {address_max:,}")
    print(f"  Mean:   {address_chars / total_rows:.2f}")

    print()


def main() -> None:
    files = [
        DATA_DIR / "train" / "train_source1.tsv",
        DATA_DIR / "train" / "train_source2.tsv",
        DATA_DIR / "train" / "train_source3.tsv",
        DATA_DIR / "test" / "test_source1.tsv",
        DATA_DIR / "test" / "test_source2.tsv",
        DATA_DIR / "test" / "test_source3.tsv",
    ]

    for path in files:
        profile_source(path)


if __name__ == "__main__":
    main()