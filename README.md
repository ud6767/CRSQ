# CRSQ for automatic ICD coding

Research code for **Rebalancing Consistency and Semantic Label Queries for Extreme Multi-Label ICD Coding**. CRSQ trains a PLM-ICD backbone on paired clinical-note views and uses separate normalization for positive and negative label pairs. The MIMIC-III-Full configurations also support frequency-dependent consistency weighting and semantic query adapters.

This repository provides the archived MIMIC-III-Full E0–E9 recipes and the MIMIC-III-Rare50 transfer-learning recipes. The experiment names, hyperparameters, and evaluation rules below refer to those source configurations. Manuscript table rows require a matching run directory with the checkpoint, configuration, and predictions.

## Setup

Run commands from the repository root. The archived dependency lock targets Python **3.11.5**. Training uses PyTorch and a CUDA GPU.

```bash
poetry install --no-root
```

Download the pretrained RoBERTa-base-PM-M3-Voc encoder using the BiLM link in [Explainable Medical Coding](https://github.com/JoakimEdin/explainable-medical-coding). Place the extracted Hugging Face model directory at `models/roberta-base-pm-m3-voc-hf/`. Alternatively, pass `model.configs.model_path=/path/to/encoder` to every training command.

Checkpoints and prediction files are saved under `models/<run-id>/` by default. The training command prints the run directory. Set `EXPERIMENT_PATH` to use a different output root. Weights & Biases logging is disabled by default. Set `WANDB_MODE=offline` for local W&B logging.

## MIMIC-III data preparation

Obtain authorized access through [PhysioNet](https://physionet.org/content/mimiciii/1.4/). This repository distributes source code. Keep the dataset, patient-level outputs, and model checkpoints in your local research storage.

1. Put `NOTEEVENTS.csv.gz`, `DIAGNOSES_ICD.csv.gz`, and `PROCEDURES_ICD.csv.gz` under `data/raw/physionet.org/files/mimiciii/1.4/`.
2. Obtain the split manifests `mimiciii_full_splits.feather` and `mimiciii_50_splits.feather` from the [upstream Explainable Medical Coding repository](https://github.com/JoakimEdin/explainable-medical-coding) and place them under `data/splits/`.
3. Run the preprocessing commands.

```bash
poetry run python -m explainable_medical_coding.data.prepare_mimiciii data/raw data/processed
poetry run python -m explainable_medical_coding.data.make_mimiciii_full
poetry run python -m scripts.build_label_frequencies
# Required for Rare50 encoder pretraining
poetry run python -m explainable_medical_coding.data.make_mimiciii_50
```

The archived split manifests have these SHA-256 digests.

| File | SHA-256 |
| --- | --- |
| `mimiciii_full_splits.feather` | `85c763780c2526bc3e5edbc45649b13f380c90bc4cfcbdd57ca16af702acfb26` |
| `mimiciii_50_splits.feather` | `7b78a5c2b1242f6114578a40943ea37ef73d0f9abd364abe462e7f6aff389d16` |

For semantic experiments, take the five files below from `data/mimic3/` in [GKI-ICD](https://github.com/xuzhang0112/GKI-ICD) and place them in `artifacts/knowledge/gki_icd/`.

- `code_description.csv`
- `code_synonym.json`
- `code_hierarchy.csv`
- `extra_code_description.csv`
- `group_description.csv`

The frequency file uses the benchmark output vocabulary in tokenizer order. Counts come exclusively from the training split.

## MIMIC-III-Full

Run E0 first to generate the shared initialization `artifacts/init_seed13.pt`. E1–E9 load this artifact before training.

```bash
poetry run python train_plm.py experiment=formal_seed13/e0_baseline gpu=0
poetry run python train_plm.py experiment=formal_seed13/e9_b3_frequency_positive_gated_semantic gpu=0
```

| Run | Configuration | Objective or model change |
| --- | --- | --- |
| E0 | `e0_baseline` | Clean supervised baseline |
| E1 | `e1_b1_paired` | Paired-view supervision |
| E2 | `e2_b2_uniform_mse` | Uniform MSE consistency |
| E3 | `e3_b3_split_normalized` | Positive/negative split-normalized MSE |
| E4 | `e4_b2_symmetric_kl` | Uniform symmetric KL |
| E5 | `e5_b1_semantic_query_adapter` | Paired-view supervision with semantic queries |
| E6 | `e6_b3_semantic_query_adapter` | Split-normalized MSE with semantic queries |
| E7 | `e7_b3_frequency_positive` | Split-normalized MSE with frequency-weighted positives |
| E8 | `e8_b3_frequency_gated_semantic` | Split-normalized MSE with frequency-gated semantic queries |
| E9 | `e9_b3_frequency_positive_gated_semantic` | Frequency-weighted positives and frequency-gated semantic queries |

The archived loss allots equal total weight to positive and negative pairs before within-group weighting. The fixed group weights are part of these recipes.

## MIMIC-III-Rare50

Generate the Rare50 CSV splits from authorized MIMIC-III data using the [KEPT data-preparation instructions](https://github.com/whaleloops/KEPT#generate-mimic-iii-rare50-data). Copy the generated `{train,dev,test}_50l.csv` files to `data/raw/rare50/` as `train_rare50.csv`, `dev_rare50.csv`, and `test_rare50.csv`. The converter expects `SUBJECT_ID`, `HADM_ID`, `TEXT`, and `LABELS` columns. The archived split has 249 training, 20 development, and 142 test notes.

```bash
poetry run python -m scripts.build_mimiciii_rare50_parquet
poetry run python train_plm.py experiment=mimiciii_rare50/seed2024_mimic50_pretrain gpu=0
# Replace PRETRAIN_RUN_ID with the run ID printed by the previous command.
poetry run python -m scripts.extract_encoder_checkpoint \
  models/PRETRAIN_RUN_ID/best_model.pt artifacts/mimiciii50_encoder_seed2024.pt
poetry run python train_plm.py experiment=mimiciii_rare50/seed2024_crsq_split gpu=0
```

The Rare50 split-normalized recipe initializes label queries from descriptions and synonyms. It uses the encoder from the selected MIMIC-III-50 checkpoint and freshly initialized Rare50 label layers. Replace `2024` with `13` or `42` in the three training and extraction paths for the other archived seeds. The separate `seed13_crsq` configuration records the frequency-weighted variant.

## Evaluation

```bash
poetry run python -m scripts.evaluate_predictions models/RUN_ID
```

The CLI reads the validation-selected model-wide threshold from `best_model.pt`. The trainer searches 100 equally spaced values in `[0, 1]` for maximum validation Micro-F1. A label is positive when its score exceeds that threshold. The CLI averages per-code F1 over the full output vocabulary, with zero for an undefined per-code F1. Micro-F1 and Micro-AUC use all outputs. Macro-AUC averages codes with both positive and negative examples. Precision@8 and Precision@15 use probability ranks.

For perturbation evaluation, generate clean and perturbed predictions with the same checkpoint. The example uses seed 1337 and the medium spelling and format settings. The analysis writes document-averaged label retention for rare and head codes to `label_retention.csv`.

```bash
poetry run python -m scripts.evaluate_frequency_robustness predict \
  --checkpoint-dir models/RUN_ID --condition clean --output reports/clean.npz
poetry run python -m scripts.evaluate_frequency_robustness predict \
  --checkpoint-dir models/RUN_ID --condition spelling --severity 0.03 --output reports/spelling.npz
poetry run python -m scripts.evaluate_frequency_robustness predict \
  --checkpoint-dir models/RUN_ID --condition format --severity 0.10 --output reports/format.npz
poetry run python -m scripts.evaluate_frequency_robustness analyze \
  --checkpoint-dir models/RUN_ID --clean-predictions reports/clean.npz \
  --prediction spelling=reports/spelling.npz --prediction format=reports/format.npz \
  --output-dir reports/robustness
```

## Tests

The included tests run on CPU with synthetic inputs and without downloading data or weights.

```bash
poetry run python -m pip install pytest
poetry run python -m pytest -q
```

## Attribution

The training backbone and MIMIC preprocessing build on [Explainable Medical Coding](https://github.com/JoakimEdin/explainable-medical-coding). Its MIT copyright notice is retained in [LICENSE](LICENSE). Semantic inputs are from [GKI-ICD](https://github.com/xuzhang0112/GKI-ICD), and Rare50 follows the [KEPT benchmark](https://aclanthology.org/2022.findings-emnlp.127/).
