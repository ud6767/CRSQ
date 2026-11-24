# Lint as: python3
"""MIMIC-III-Rare50 generated with the Yang et al. (2022) protocol."""

import datasets
import polars as pl

from explainable_medical_coding.utils.settings import REPOSITORY_PATH

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "MIMIC-III-Rare50: 50 rare ICD-9-CM diagnosis codes selected from MIMIC-III."
_URL = REPOSITORY_PATH / "data" / "processed" / "mimiciii_rare50"
_URLS = {
    "train": _URL / "train.parquet",
    "validation": _URL / "dev.parquet",
    "test": _URL / "test.parquet",
}


class MIMIC_III_Rare50_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-III-Rare50."""


class MIMIC_III_Rare50(datasets.GeneratorBasedBuilder):
    """MIMIC-III-Rare50 with the repository's standard Dataset interface."""

    BUILDER_CONFIGS = [
        MIMIC_III_Rare50_Config(
            name="mimiciii-rare50",
            version=datasets.Version("1.0.0"),
            description="Rare50 diagnosis-code benchmark",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "subject_id": datasets.Value("int64"),
                    "_id": datasets.Value("int64"),
                    "note_type": datasets.Value("string"),
                    "note_id": datasets.Value("string"),
                    "note_subtype": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "diagnosis_codes": datasets.Sequence(datasets.Value("string")),
                    "diagnosis_code_type": datasets.Value("string"),
                    "procedure_codes": datasets.Sequence(datasets.Value("string")),
                    "procedure_code_type": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("Generating Rare50 examples from %s", filepath)
        dataframe = pl.read_parquet(filepath)
        for row in dataframe.to_dicts():
            yield row["_id"], row
