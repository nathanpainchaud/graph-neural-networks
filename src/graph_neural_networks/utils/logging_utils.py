from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torchmetrics import MetricCollection

from graph_neural_networks.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally, saves:
        - Number of model parameters

    Args:
        object_dict: A dictionary containing the following objects: `"cfg"`: a DictConfig object containing the main
            config, `"model"`: the Lightning model, `"trainer"`: the Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def pad_keys(
    mapping: Mapping[str, Any],
    prefix: str | None = None,
    postfix: str | None = None,
    exclude: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pads the keys of a mapping with a combination of prefix/postfix.

    Args:
        mapping: Mapping with string keys for which to add a prefix to the keys.
        prefix: Prefix to prepend to the current keys in the mapping.
        postfix: Postfix to append to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Dictionary where the keys have been prepended with `prefix` / appended with `postfix`.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    if prefix is None:
        prefix = ""
    if postfix is None:
        postfix = ""

    return {f"{prefix}{k}{postfix}" if k not in exclude else k: v for k, v in mapping.items()}


def split_scalar_nonscalar_metrics(
    metrics: MetricCollection,
) -> tuple[MetricCollection | None, MetricCollection | None]:
    """Splits a MetricCollection into scalar and non-scalar metrics.

    Args:
        metrics: MetricCollection to split.

    Returns:
        A tuple containing two MetricCollections: the first with scalar metrics, the second with non-scalar metrics.
    """
    # Infer whether a metric is scalar or not based on whether it implements `higher_is_better` attribute.
    # This might not be perfect, but it works for most common cases.
    scalar_metrics = {}
    nonscalar_metrics = {}

    for tag, metric in (metrics or {}).items():
        if metric.higher_is_better is not None:
            scalar_metrics[tag] = metric
        else:
            nonscalar_metrics[tag] = metric

    return (
        MetricCollection(scalar_metrics) if scalar_metrics else None,
        MetricCollection(nonscalar_metrics) if nonscalar_metrics else None,
    )


def log_nonscalar_metrics(logger: Logger, metrics: MetricCollection) -> None:
    """Log non-scalar metrics as figures, if the logger supports it.

    Args:
        logger: Logger to log to.
        metrics: MetricCollection containing the non-scalar metrics to log.

    Raises:
        NotImplementedError: If support for non-scalar metrics has not been implemented for the given logger.
    """
    # Disable matplotlib text rendering with LaTeX, to avoid errors or warnings about missing
    # on systems with LaTeX, but w/o all the expected packages and fonts
    with plt.rc_context({"text.usetex": False}):
        # Plot non-scalar metrics as figures
        plots = metrics.plot()

        match logger:
            case WandbLogger():
                wandb_run = logger.experiment
                for tag, (fig_, ax_) in zip(metrics.keys(), plots, strict=False):  # noqa: B007
                    wandb_run.log({tag: fig_})
            case None:
                pass  # not logging if no logger is configured
            case _:
                raise NotImplementedError(
                    f"Logging non-scalar metrics is only implemented for wandb logger, found {type(logger)}."
                )

        plt.close("all")  # avoid memory leaks from figures left opened


def create_predictions_dataframe(
    predictions: np.ndarray,
    batch_indices: Sequence[int] | np.ndarray | None = None,
    output_labels: Sequence[str] | None = None,
    samplewise_op: Literal["softmax", "argmax"] | None = None,
) -> pd.DataFrame:
    """Create a DataFrame from predictions.

    Args:
        predictions: Array of model predictions.
            (n_samples,) for regression or (n_samples, n_classes) for classification.
        batch_indices: Batch indices corresponding to the predictions. If None, batch indices are assumed to be
            continuous from 0 to n_samples-1.
        output_labels: Sequence of label names corresponding to prediction columns, for models that return multiple
            values per sample (e.g. class logits). If None, uses numeric indices as column names.
        samplewise_op: Operation to apply to predictions on a per-sample basis before creating the DataFrame.
            - "softmax": Apply softmax along the class dimension.
            - "argmax": Take the argmax along the class dimension.
            - None: Use predictions as is.

    Returns:
        DataFrame containing the predictions and optionally batch indices.
    """
    # Apply samplewise operation if specified
    proc_predictions = predictions
    match samplewise_op:
        case "softmax":
            proc_predictions = scipy.special.softmax(proc_predictions, axis=1)
        case "argmax":
            proc_predictions = np.argmax(proc_predictions, axis=1)
        case None:
            # No operation, use predictions as is
            pass
        case _:
            raise ValueError(f"Unsupported samplewise operation: {samplewise_op}")

    # Create DataFrame based on prediction dimensionality
    if proc_predictions.ndim == 1:
        # Regression: single value per sample
        data = {"prediction": proc_predictions}
    else:
        # Classification: multiple values per sample (i.e. class probabilities)
        output_labels = output_labels or [str(i) for i in range(proc_predictions.shape[1])]
        data = {output_label: proc_predictions[:, i] for i, output_label in enumerate(output_labels)}

    # Add batch indices if provided, otherwise create default continuous indices
    if batch_indices is not None:
        data["batch_idx"] = batch_indices
    else:
        data["batch_idx"] = list(range(len(proc_predictions)))

    return pd.DataFrame(data)


def log_dataframe(
    df: pd.DataFrame,
    logger: Logger | list[Logger],
    table_name: str,
) -> None:
    """Log DataFrame to experiment tracker.

    Currently only supports WandbLogger. For other loggers, this function does nothing.

    Args:
        df: DataFrame to log.
        logger: Logger or list of loggers to log to.
        table_name: Name for the table in the experiment tracker.
    """
    # Handle both single logger and list of loggers
    loggers = logger if isinstance(logger, list) else [logger]

    for logger_instance in loggers:
        # Log as a WandB Table for WandbLogger
        if isinstance(logger_instance, WandbLogger):
            import wandb  # noqa: PLC0415

            # Create WandB Table from DataFrame
            table = wandb.Table(dataframe=df)
            wandb_run = logger_instance.experiment
            wandb_run.log({table_name: table})
