"""Convert the reproducible Rare50 CSV splits to the repository schema."""

from pathlib import Path
from collections import Counter
import csv

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "rare50"
OUT = ROOT / "data" / "processed" / "mimiciii_rare50"


def convert(source: Path, destination: Path) -> None:
    frame = pl.read_csv(source)
    labels = pl.col("LABELS").str.split(";").list.eval(pl.element().str.strip_chars())
    frame = frame.with_columns(
        labels.alias("diagnosis_codes"),
        pl.lit([]).cast(pl.List(pl.Utf8)).alias("procedure_codes"),
        pl.lit("Discharge summary").alias("note_type"),
        pl.col("HADM_ID").cast(pl.Utf8).alias("note_id"),
        pl.lit("report").alias("note_subtype"),
        pl.lit("icd9cm").alias("diagnosis_code_type"),
        pl.lit("icd9pcs").alias("procedure_code_type"),
        pl.col("TEXT").alias("text"),
        pl.col("SUBJECT_ID").cast(pl.Int64).alias("subject_id"),
        pl.col("HADM_ID").cast(pl.Int64).alias("_id"),
    ).select(
        [
            "subject_id",
            "_id",
            "note_type",
            "note_id",
            "note_subtype",
            "text",
            "diagnosis_codes",
            "diagnosis_code_type",
            "procedure_codes",
            "procedure_code_type",
        ]
    )
    frame = frame.with_columns(
        pl.col("diagnosis_codes").list.unique(maintain_order=True)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination)
    print(f"wrote {destination}: {frame.shape}")


if __name__ == "__main__":
    convert(RAW / "train_rare50.csv", OUT / "train.parquet")
    convert(RAW / "dev_rare50.csv", OUT / "dev.parquet")
    convert(RAW / "test_rare50.csv", OUT / "test.parquet")
    train = pl.read_csv(RAW / "train_rare50.csv")
    counts = Counter(
        code.strip()
        for value in train["LABELS"]
        for code in str(value).split(";")
        if code.strip()
    )
    frequency_path = ROOT / "artifacts" / "mimiciii_rare50_label_frequency.csv"
    frequency_path.parent.mkdir(parents=True, exist_ok=True)
    with frequency_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target", "train_frequency"])
        writer.writeheader()
        for code in sorted(counts):
            writer.writerow({"target": code, "train_frequency": counts[code]})
    print(f"wrote {frequency_path}: {len(counts)} labels")
