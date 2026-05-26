#!/usr/bin/env python3
"""
Create a reproducible 20,000-row sample from the original Kaggle vehicles.csv.

Place the original Kaggle file at data/vehicles.csv, then run:
    python scripts/create_random_sample.py

The script writes:
    data/vehicles_sample_20000.csv
"""

from pathlib import Path
import csv
import random

INPUT_PATH = Path("data/vehicles.csv")
OUTPUT_PATH = Path("data/vehicles_sample_20000.csv")
SAMPLE_SIZE = 20_000
RANDOM_SEED = 42


def create_sample(input_path: Path, output_path: Path, sample_size: int, random_seed: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find {input_path}. Download vehicles.csv from Kaggle first.")

    rng = random.Random(random_seed)
    reservoir = []
    valid_rows = 0
    skipped_rows = 0

    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)

        for total_rows, row in enumerate(reader, start=1):
            if len(row) != n_cols:
                skipped_rows += 1
                continue
            valid_rows += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = rng.randrange(valid_rows)
                if j < sample_size:
                    reservoir[j] = row
            if total_rows % 50_000 == 0:
                print(f"Read={total_rows:,}, valid={valid_rows:,}, skipped={skipped_rows:,}")

    rng.shuffle(reservoir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(reservoir)

    print(f"Saved {len(reservoir):,} sampled rows to {output_path}")
    print(f"Valid rows={valid_rows:,}, skipped broken rows={skipped_rows:,}")


if __name__ == "__main__":
    create_sample(INPUT_PATH, OUTPUT_PATH, SAMPLE_SIZE, RANDOM_SEED)
