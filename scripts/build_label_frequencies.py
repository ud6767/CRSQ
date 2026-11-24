"""Write training-label counts in MIMIC-III-Full tokenizer order."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import polars as pl


def count_labels(path: Path) -> Counter[str]:
    rows = pl.read_parquet(path, columns=["diagnosis_codes", "procedure_codes"])
    counts = Counter()
    for diagnosis, procedure in rows.iter_rows():
        codes = set((diagnosis or []) + (procedure or []))
        counts.update(codes)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/mimiciii_full"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/frequency/label_frequency.csv"))
    args = parser.parse_args()
    train = count_labels(args.data_dir / "train.parquet")
    # The existing data loader fits the vocabulary over all benchmark splits.
    vocabulary = train.keys() | count_labels(args.data_dir / "val.parquet").keys()
    vocabulary |= count_labels(args.data_dir / "test.parquet").keys()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("code", "train_frequency"))
        writer.writerows((code, train[code]) for code in sorted(vocabulary))
    print(f"Wrote {len(vocabulary)} code frequencies to {args.output}")


if __name__ == "__main__":
    main()
