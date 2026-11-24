# ruff: noqa: E402
import logging
import hashlib
import math
from collections import Counter
from pathlib import Path

# load environment variables
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import hydra
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import explainable_medical_coding.config.factories as factories
from explainable_medical_coding.utils.loaders import (
    load_trained_model,
)
from explainable_medical_coding.utils.polarity import (
    PolarityGuidanceGenerator,
    load_mimiciii_diagnosis_descriptions,
)
from explainable_medical_coding.utils.tokenizer import (
    TargetTokenizer,
    tokenize_texts_with_chunk_special_tokens,
)
from explainable_medical_coding.utils.analysis import predict
from explainable_medical_coding.utils.data_helper_functions import (
    create_targets_column,
    filter_unknown_targets,
    get_unique_targets,
)
from explainable_medical_coding.utils.seed import set_seed
from explainable_medical_coding.utils.settings import TARGET_COLUMN, TEXT_COLUMN
from explainable_medical_coding.utils.tensor import deterministic, set_gpu
from explainable_medical_coding.utils.paired_augmentation import (
    augment_training_text,
)
from explainable_medical_coding.trainer.consistency import (
    ConsistencyDiagnosticsTracker,
)
from explainable_medical_coding.utils.semantic_queries import (
    initialize_semantic_label_queries,
    initialize_semantic_query_adapter,
)

LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


def model_state_sha256(
    model: torch.nn.Module,
    keys: set[str] | None = None,
) -> str:
    """Hash an initialized model so matched formal runs can prove parity."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if keys is not None and name not in keys:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        # NumPy handles both ordinary tensors and 0-D parameters (for example
        # the semantic adapter's scalar alpha) without changing byte order.
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@hydra.main(
    version_base=None,
    config_path="explainable_medical_coding/config",
    config_name="config",
)
def main(cfg: OmegaConf) -> None:
    if cfg.deterministic:
        deterministic()

    set_seed(cfg.seed)
    device = set_gpu(cfg)

    target_columns = list(cfg.data.target_columns)
    dataset_path = Path(cfg.data.dataset_path)
    model_path = Path(cfg.load_model) if cfg.load_model is not None else None
    dataset = load_dataset(str(dataset_path), trust_remote_code=True)
    paired_training = bool(
        OmegaConf.select(cfg, "paired_training.enabled", default=False)
    )

    text_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.configs.model_path,
    )
    max_input_length = int(cfg.data.max_length)
    add_chunk_special_tokens = bool(
        OmegaConf.select(cfg, "data.add_chunk_special_tokens", default=False)
    )
    polarity_guidance = bool(
        OmegaConf.select(cfg, "data.polarity_guidance", default=False)
    )
    if add_chunk_special_tokens:
        chunk_size = int(cfg.model.configs.chunk_size)
        LOGGER.info(
            "Tokenizing each %d-token encoder chunk with its own special tokens",
            chunk_size,
        )

        def tokenize_texts(texts, return_offsets_mapping=False):
            return tokenize_texts_with_chunk_special_tokens(
                texts=texts,
                text_tokenizer=text_tokenizer,
                max_length=max_input_length,
                chunk_size=chunk_size,
                return_offsets_mapping=return_offsets_mapping,
            )

    else:
        LOGGER.info("Using document-level special tokens (baseline tokenization)")

        def tokenize_texts(texts, return_offsets_mapping=False):
            return text_tokenizer(
                texts,
                return_length=True,
                truncation=True,
                max_length=max_input_length,
                return_offsets_mapping=return_offsets_mapping,
            )

    dataset = dataset.map(
        lambda x: create_targets_column(x, target_columns),
        desc="Creating targets column",
    )
    known_targets = set(get_unique_targets(dataset))
    dataset = dataset.map(
        lambda x: filter_unknown_targets(x, known_targets=known_targets),
        desc="Filter unknown targets",
    )
    dataset = dataset.filter(
        lambda x: len(x[TARGET_COLUMN]) > 0, desc="Filtering empty targets"
    )

    autoregressive = bool(cfg.model.autoregressive)
    target_tokenizer = TargetTokenizer(autoregressive=autoregressive)
    if model_path is None:
        unique_targets = get_unique_targets(dataset)
        target_tokenizer.fit(unique_targets)
    else:
        LOGGER.info("Loading Tokenizer from model_path")
        target_tokenizer.load(model_path / "target_tokenizer.json")

    # convert targets to target ids
    dataset = dataset.map(
        lambda x: {"target_ids": target_tokenizer(x[TARGET_COLUMN])},
        desc="Converting targets to target ids",
    )

    label_frequencies = Counter()
    for target_names in dataset["train"][TARGET_COLUMN]:
        label_frequencies.update(set(target_names))
    frequency_tensor = torch.tensor(
        [label_frequencies.get(code, 0) for code in target_tokenizer.target_names()],
        dtype=torch.long,
    )

    for split_name, config_key in (
        ("train", "data.train_max_samples"),
        ("validation", "data.validation_max_samples"),
        ("test", "data.test_max_samples"),
    ):
        max_samples = OmegaConf.select(cfg, config_key, default=None)
        if max_samples is not None:
            sample_count = min(int(max_samples), len(dataset[split_name]))
            LOGGER.warning(
                "Restricting %s to %d rows for an engineering run",
                split_name,
                sample_count,
            )
            dataset[split_name] = dataset[split_name].select(range(sample_count))

    if paired_training:
        augmentation_seed = int(cfg.paired_training.augmentation_seed)
        LOGGER.info(
            "Building deterministic paired training views with seed %d",
            augmentation_seed,
        )

        def augment_training_row(row):
            record_key = row.get("note_id", row["_id"])
            return augment_training_text(
                text=row[TEXT_COLUMN],
                record_key=record_key,
                seed=augmentation_seed,
            )

        dataset["train"] = dataset["train"].map(
            augment_training_row,
            desc="Building paired clean/aug training views",
        )
    formatted_columns = ["input_ids", "length", "attention_mask", "target_ids"]
    if polarity_guidance:
        description_path = Path(
            OmegaConf.select(
                cfg,
                "data.polarity_diagnosis_description_path",
                default=(
                    "data/raw/physionet.org/files/mimiciii/1.4/"
                    "D_ICD_DIAGNOSES.csv.gz"
                ),
            )
        )
        LOGGER.info(
            "Generating training-only explicit-negation guidance from %s",
            description_path,
        )
        polarity_generator = PolarityGuidanceGenerator(
            target_tokenizer=target_tokenizer,
            code_to_description=load_mimiciii_diagnosis_descriptions(
                description_path
            ),
        )

        def tokenize_training_texts_with_polarity(batch):
            tokenized_batch = tokenize_texts(
                batch[TEXT_COLUMN], return_offsets_mapping=True
            )
            offset_mappings = tokenized_batch.pop("offset_mapping")
            guidance = [
                polarity_generator(
                    text=text,
                    assigned_codes=assigned_codes,
                    offset_mapping=offset_mapping,
                )
                for text, assigned_codes, offset_mapping in zip(
                    batch[TEXT_COLUMN],
                    batch[TARGET_COLUMN],
                    offset_mappings,
                )
            ]
            tokenized_batch["polarity_label_ids"] = [
                item["polarity_label_ids"] for item in guidance
            ]
            tokenized_batch["polarity_token_ids"] = [
                item["polarity_token_ids"] for item in guidance
            ]
            return tokenized_batch

        dataset["train"] = dataset["train"].map(
            tokenize_training_texts_with_polarity,
            batched=True,
            num_proc=8,
            batch_size=1_000,
            desc="Tokenizing text and generating explicit-negation guidance",
        )
        for split_name in list(dataset.keys()):
            if split_name == "train":
                continue
            dataset[split_name] = dataset[split_name].map(
                lambda x: {
                    **tokenize_texts(x[TEXT_COLUMN]),
                    "polarity_label_ids": [[] for _ in x[TEXT_COLUMN]],
                    "polarity_token_ids": [[] for _ in x[TEXT_COLUMN]],
                },
                batched=True,
                num_proc=8,
                batch_size=1_000,
                desc=f"Tokenizing {split_name} text",
            )
        formatted_columns.extend(["polarity_label_ids", "polarity_token_ids"])
    else:
        dataset = dataset.map(
            lambda x: tokenize_texts(x[TEXT_COLUMN]),
            batched=True,
            num_proc=8,
            batch_size=1_000,
            desc="Tokenizing text",
        )
    if paired_training:
        def tokenize_augmented_texts(batch):
            tokenized = tokenize_texts(batch["augmented_text"])
            return {
                "augmented_input_ids": tokenized["input_ids"],
                "augmented_attention_mask": tokenized["attention_mask"],
                "augmented_length": tokenized["length"],
            }

        dataset["train"] = dataset["train"].map(
            tokenize_augmented_texts,
            batched=True,
            num_proc=8,
            batch_size=1_000,
            desc="Tokenizing augmented training text",
        )
    for split_name in dataset.keys():
        split_columns = list(formatted_columns)
        if paired_training and split_name == "train":
            split_columns.extend(
                [
                    "augmented_input_ids",
                    "augmented_attention_mask",
                    "augmented_length",
                ]
            )
        dataset[split_name].set_format(type="torch", columns=split_columns)

    lookups = factories.get_lookups(
        dataset=dataset,
        text_tokenizer=text_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    LOGGER.info(lookups.data_info)

    decision_boundary = 0.5
    if model_path is None:
        model = factories.get_model(config=cfg.model, data_info=lookups.data_info)
    else:
        LOGGER.info("Loading Model from model_path")
        saved_config = OmegaConf.load(model_path / "config.yaml")
        model, decision_boundary = load_trained_model(
            model_path,
            saved_config,
            pad_token_id=text_tokenizer.pad_token_id,
            device=device,
        )

    encoder_initialization_checkpoint = OmegaConf.select(
        cfg, "encoder_initialization_checkpoint", default=None
    )
    if encoder_initialization_checkpoint is not None:
        payload = torch.load(
            encoder_initialization_checkpoint, map_location="cpu", weights_only=False
        )
        encoder_state = {
            key: value for key, value in payload.get("model", payload).items()
            if key.startswith("roberta_encoder.")
        }
        expected_keys = {
            key for key in model.state_dict() if key.startswith("roberta_encoder.")
        }
        if not encoder_state or encoder_state.keys() != expected_keys:
            raise ValueError("The checkpoint must contain the complete RoBERTa encoder")
        model.load_state_dict(encoder_state, strict=False)
        LOGGER.info("Loaded Rare50 encoder from %s", encoder_initialization_checkpoint)

    initialization_checkpoint = OmegaConf.select(
        cfg, "initialization_checkpoint", default=None
    )
    target_names = list(target_tokenizer.target_names())
    if initialization_checkpoint is not None:
        initialization_checkpoint = Path(initialization_checkpoint)
        payload = torch.load(
            initialization_checkpoint, map_location="cpu", weights_only=False
        )
        if list(payload["target_names"]) != target_names:
            raise ValueError(
                "Formal initialization target vocabulary/order does not match"
            )
        if int(payload["num_classes"]) != int(lookups.data_info["num_classes"]):
            raise ValueError("Formal initialization class count does not match")
        semantic_adapter_enabled = bool(
            OmegaConf.select(
                cfg, "model.configs.semantic_query_adapter", default=False
            )
        )
        if semantic_adapter_enabled:
            incompatible = model.load_state_dict(payload["model"], strict=False)
            allowed_prefix = "label_wise_attention.semantic_"
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_prefix)
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "Semantic adapter initialization mismatch: "
                    f"missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        else:
            model.load_state_dict(payload["model"], strict=True)
        LOGGER.info(
            "Loaded formal initialization from %s", initialization_checkpoint
        )

    initial_fingerprint = model_state_sha256(model)
    LOGGER.info("Initial model SHA256: %s", initial_fingerprint)
    if initialization_checkpoint is not None:
        expected_fingerprint = str(payload["model_sha256"])
        if semantic_adapter_enabled:
            base_fingerprint = model_state_sha256(
                model, keys=set(payload["model"].keys())
            )
            LOGGER.info("Base initialization SHA256: %s", base_fingerprint)
        else:
            base_fingerprint = initial_fingerprint
        if base_fingerprint != expected_fingerprint:
            raise ValueError(
                "Loaded formal initialization fingerprint does not match"
            )

    save_initialization_checkpoint = OmegaConf.select(
        cfg, "save_initialization_checkpoint", default=None
    )
    if save_initialization_checkpoint is not None:
        save_initialization_checkpoint = Path(save_initialization_checkpoint)
        save_initialization_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        initialization_payload = {
            "model": model.state_dict(),
            "model_sha256": initial_fingerprint,
            "target_names": target_names,
            "num_classes": int(lookups.data_info["num_classes"]),
            "seed": int(cfg.seed),
        }
        if save_initialization_checkpoint.exists():
            existing_payload = torch.load(
                save_initialization_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if (
                str(existing_payload.get("model_sha256"))
                != initial_fingerprint
                or list(existing_payload.get("target_names", []))
                != target_names
            ):
                raise ValueError(
                    "Existing formal initialization checkpoint is incompatible"
                )
            LOGGER.info(
                "Verified existing formal initialization at %s",
                save_initialization_checkpoint,
            )
        else:
            torch.save(initialization_payload, save_initialization_checkpoint)
            LOGGER.info(
                "Saved formal initialization to %s",
                save_initialization_checkpoint,
            )

    model.to(device)
    if bool(OmegaConf.select(cfg, "semantic_query_adapter.enabled", default=False)):
        if not bool(
            OmegaConf.select(
                cfg, "model.configs.semantic_query_adapter", default=False
            )
        ):
            raise ValueError(
                "semantic_query_adapter.enabled requires "
                "model.configs.semantic_query_adapter=true"
            )
        adapter_report = initialize_semantic_query_adapter(
            model=model,
            tokenizer=text_tokenizer,
            target_names=target_tokenizer.target_names(),
            description_path=cfg.semantic_query_adapter.description_path,
            synonym_path=cfg.semantic_query_adapter.synonym_path,
            hierarchy_relation_path=(
                cfg.semantic_query_adapter.hierarchy_relation_path
            ),
            hierarchy_group_path=cfg.semantic_query_adapter.hierarchy_group_path,
            hierarchy_extra_description_path=(
                cfg.semantic_query_adapter.hierarchy_extra_description_path
            ),
            max_synonyms=int(cfg.semantic_query_adapter.max_synonyms),
            batch_size=int(cfg.semantic_query_adapter.batch_size),
            max_length=int(cfg.semantic_query_adapter.max_length),
        )
        LOGGER.info("Semantic query adapter: %s", adapter_report.to_dict())
        if bool(
            OmegaConf.select(
                cfg, "model.configs.semantic_query_frequency_gate", default=False
            )
        ):
            attention = getattr(model, "label_wise_attention", None)
            attention.set_semantic_query_frequency_gate(
                frequency_tensor,
                tail_alpha=float(
                    OmegaConf.select(
                        cfg, "semantic_query_adapter.tail_alpha", default=0.2
                    )
                ),
                torso_alpha=float(
                    OmegaConf.select(
                        cfg, "semantic_query_adapter.torso_alpha", default=0.1
                    )
                ),
                head_alpha=float(
                    OmegaConf.select(
                        cfg, "semantic_query_adapter.head_alpha", default=0.05
                    )
                ),
                tail_max_frequency=int(
                    OmegaConf.select(
                        cfg,
                        "semantic_query_adapter.tail_max_frequency",
                        default=10,
                    )
                ),
                torso_max_frequency=int(
                    OmegaConf.select(
                        cfg,
                        "semantic_query_adapter.torso_max_frequency",
                        default=100,
                    )
                ),
            )
            LOGGER.info("Enabled frequency-gated semantic query residuals")
        LOGGER.info(
            "Semantic-adapted model SHA256: %s", model_state_sha256(model)
        )
    if bool(OmegaConf.select(cfg, "semantic_query.enabled", default=False)):
        description_path = OmegaConf.select(
            cfg, "semantic_query.description_path", default=None
        )
        if not description_path:
            raise ValueError(
                "semantic_query.description_path is required when semantic queries are enabled"
            )
        semantic_report = initialize_semantic_label_queries(
            model=model,
            tokenizer=text_tokenizer,
            target_names=target_tokenizer.target_names(),
            description_path=description_path,
            synonym_path=OmegaConf.select(
                cfg, "semantic_query.synonym_path", default=None
            ),
            use_synonyms=bool(
                OmegaConf.select(cfg, "semantic_query.use_synonyms", default=False)
            ),
            synonym_weight=float(
                OmegaConf.select(cfg, "semantic_query.synonym_weight", default=0.3)
            ),
            max_synonyms=int(
                OmegaConf.select(cfg, "semantic_query.max_synonyms", default=5)
            ),
            batch_size=int(
                OmegaConf.select(cfg, "semantic_query.batch_size", default=128)
            ),
            max_length=int(
                OmegaConf.select(cfg, "semantic_query.max_length", default=128)
            ),
            label_frequencies=frequency_tensor,
            frequency_gamma=float(
                OmegaConf.select(cfg, "semantic_query.frequency_gamma", default=0.5)
            ),
            frequency_weight_max=float(
                OmegaConf.select(
                    cfg, "semantic_query.frequency_weight_max", default=5.0
                )
            ),
        )
        LOGGER.info("Semantic query initialization: %s", semantic_report.to_dict())

    if bool(OmegaConf.select(cfg, "tail_residual.enabled", default=False)):
        if model_path is None:
            raise ValueError("Tail residual training requires load_model")
        model.configure_tail_residual(
            frequencies=frequency_tensor,
            mode=str(OmegaConf.select(cfg, "tail_residual.mode")),
            tail_max_frequency=int(
                OmegaConf.select(cfg, "tail_residual.tail_max_frequency", default=10)
            ),
            max_scale=float(
                OmegaConf.select(cfg, "tail_residual.max_scale", default=0.5)
            ),
        )
        model.freeze_for_tail_residual()
        trainable = {
            name: parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        LOGGER.info(
            "Configured %s Tail residual: labels=%d trainable=%s total=%d",
            OmegaConf.select(cfg, "tail_residual.mode"),
            model.tail_residual.tail_indices.numel(),
            trainable,
            sum(trainable.values()),
        )
    # model = torch.compile(model)

    if cfg.distillation:
        if model_path is None:
            raise ValueError("Distillation requires a pre-trained model")
        dataset = dataset.sort("length")
        model.eval()
        dataset = dataset.map(
            lambda x: {
                "teacher_logits": predict(
                    model,
                    x["input_ids"],
                    device=device,
                    return_logits=True,
                    pad_id=text_tokenizer.pad_token_id,
                ),
            },
            desc="Adding teacher logits",
            batched=True,
            batch_size=64,
        )
        model.train()

    loss_function = factories.get_loss_function(config=cfg.loss)

    dataloaders = factories.get_dataloaders(
        config=cfg.dataloader,
        dataset=dataset,
        target_tokenizer=lookups.target_tokenizer,
        pad_token_id=lookups.data_info["pad_token_id"],
    )

    metric_collections = factories.get_metric_collections(
        config=cfg.metrics,
        number_of_classes=lookups.data_info["num_classes"],
        split2code_indices=lookups.split2code_indices,
        autoregressive=cfg.model.autoregressive,
    )

    optimizer = factories.get_optimizer(config=cfg.optimizer, model=model)
    accumulate_grad_batches = int(
        max(cfg.dataloader.batch_size / cfg.dataloader.max_batch_size, 1)
    )
    num_training_steps = (
        math.ceil(len(dataloaders["train"]) / accumulate_grad_batches)
        * cfg.trainer.epochs
    )
    max_optimizer_steps = OmegaConf.select(
        cfg, "trainer.max_optimizer_steps", default=None
    )
    if max_optimizer_steps is not None:
        num_training_steps = min(num_training_steps, int(max_optimizer_steps))
    consistency_tracker = None
    if paired_training:
        consistency_tracker = ConsistencyDiagnosticsTracker(
            label_frequencies=frequency_tensor,
            total_steps=num_training_steps,
            decision_boundary=float(decision_boundary),
            easy_threshold=float(cfg.paired_training.easy_negative_threshold),
            checkpoints=tuple(cfg.paired_training.diagnostic_fractions),
        )
    lr_scheduler = factories.get_lr_scheduler(
        config=cfg.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )
    callbacks = factories.get_callbacks(config=cfg.callbacks)
    trainer_class = factories.get_trainer(name=cfg.trainer.name)
    trainer = trainer_class(
        config=cfg,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metric_collections=metric_collections,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        lookups=lookups,
        accumulate_grad_batches=accumulate_grad_batches,
        consistency_tracker=consistency_tracker,
    ).to(device)

    trainer.fit()


if __name__ == "__main__":
    main()
