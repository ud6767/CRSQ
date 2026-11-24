import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

# load environment variables
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

load_dotenv(find_dotenv())

experiment_path = os.getenv("EXPERIMENT_PATH", "models")

FORMATTING_PATTERN = r"\[([^\]]+)\]"
FORMATTING_REGEX = re.compile(FORMATTING_PATTERN)


def length_of_formatting(string: str):
    return sum(
        len(s) + 2 for s in FORMATTING_REGEX.findall(string)
    )  # plus 2 for parenthesis


def length_without_formatting(string: str):
    return len(string) - length_of_formatting(string)


def source_string(source):
    return f"{source[:18]}.." if len(source) > 20 else f"{source}"


class BaseCallback:
    def __init__(self):
        pass

    def on_initialisation_end(self, trainer=None):
        pass

    def on_train_begin(self, trainer=None):
        pass

    def on_train_end(self, trainer=None):
        pass

    def on_val_begin(self, trainer=None):
        pass

    def on_val_end(self, trainer=None):
        pass

    def on_epoch_begin(self, trainer=None):
        pass

    def on_epoch_end(self, trainer=None):
        pass

    def on_batch_begin(self, trainer=None):
        pass

    def on_batch_end(self, trainer=None):
        pass

    def on_fit_begin(self, trainer=None):
        pass

    def on_fit_end(self, trainer=None):
        pass

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        pass

    def on_end(self, trainer=None):
        pass


class WandbCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.wandb = None

    def extract_tags(self, trainer) -> list[str]:
        tags = []
        tags.append(trainer.config.model.name)
        tags.append(Path(trainer.config.data.dataset_path).stem)
        return tags

    def on_initialisation_end(self, trainer=None):
        mode = "disabled" if trainer.config.debug else os.getenv("WANDB_MODE", "disabled")
        if mode == "disabled":
            if not trainer.config.debug:
                trainer.experiment_path = Path(experiment_path) / uuid4().hex[:12]
                trainer.experiment_path.mkdir(exist_ok=False, parents=True)
            print(f"Run directory: {trainer.experiment_path}")
            return
        import wandb

        self.wandb = wandb

        wandb_cfg = OmegaConf.to_container(
            trainer.config, resolve=True, throw_on_missing=True
        )
        tags = self.extract_tags(trainer)

        wandb_cfg["num_parameters"] = sum(p.numel() for p in trainer.model.parameters())
        wandb_cfg["num_trainable_parameters"] = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )

        if trainer.lookups is not None:
            wandb_cfg["data_info"] = trainer.lookups.data_info
            OmegaConf.save(trainer.config, trainer.experiment_path / "config.yaml")

        wandb.init(
            config=wandb_cfg,
            settings=wandb.Settings(start_method="thread"),
            tags=tags,
            name=trainer.config.name,
            mode=mode,
            **self.config,
        )
        wandb.watch(trainer.model)

        trainer.experiment_path = Path(experiment_path) / wandb.run.id
        trainer.experiment_path.mkdir(exist_ok=False, parents=True)
        print(f"Run directory: {trainer.experiment_path}")

    def log_dict(
        self,
        nested_dict: dict[str, Any],
        epoch: int,
    ) -> None:
        if self.wandb is None:
            return
        nested_dict["epoch"] = epoch
        self.wandb.log(nested_dict)

    def on_end(self, trainer=None):
        if self.wandb is not None:
            self.wandb.finish()


class SaveBestModelCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.prev_best = None
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric

    def on_epoch_end(self, trainer=None):
        best_metric = trainer.metric_collections[self.split_name].get_best_metric(
            self.metric_name
        )
        if self.prev_best is None or best_metric != self.prev_best:
            self.prev_best = best_metric
            trainer.save_checkpoint("best_model.pt")
            print("Saved best model")

    def on_fit_end(self, trainer=None):
        trainer.load_checkpoint("best_model.pt")
        print("Loaded best model")


class SaveLastModelCallback(BaseCallback):
    """Keep an epoch-boundary checkpoint with full continuation state."""

    def __init__(self, config: OmegaConf):
        super().__init__()

    def on_epoch_end(self, trainer=None):
        trainer.save_checkpoint("last_model.pt")


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric
        self.patience = config.patience
        self.min_delta = float(OmegaConf.select(config, "min_delta", default=0.0))
        self.mode = str(OmegaConf.select(config, "mode", default="max"))
        self.counter = 0
        self.prev_best = None

    def on_epoch_end(self, trainer=None):
        """Stop after the configured number of epochs without a material improvement.

        Args:
            trainer (Trainer, optional): Trainer class. Defaults to None.
        """
        metric = trainer.current_val_results[self.split_name][self.metric_name]
        current = (
            float(metric.detach().cpu().item())
            if torch.is_tensor(metric)
            else float(metric)
        )
        if self.prev_best is None:
            improved = True
        elif self.mode == "max":
            improved = current > self.prev_best + self.min_delta
        elif self.mode == "min":
            improved = current < self.prev_best - self.min_delta
        else:
            raise ValueError(f"Unsupported early-stopping mode: {self.mode}")

        if improved:
            self.prev_best = current
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            trainer.stop_training = True
            print(
                f"Early stopping: {self.counter} epochs without at least "
                f"{self.min_delta:g} improvement for {self.metric_name}"
            )
