import inspect
import types
from abc import ABC
from collections.abc import Sequence
from typing import Any, Literal

import torch
from lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.datasets import FakeDataset
from torchmetrics import MeanMetric, MetricCollection, MetricTracker

from graph_neural_networks.utils import RankedLogger, pad_keys
from graph_neural_networks.utils.utils import import_from_module

log = RankedLogger(__name__, rank_zero_only=True)


class MetricTrackingLitModule(LightningModule, ABC):
    """A `LightningModule` that provides the boilerplate code for updating/computing/logging metrics during training.

    This module is designed to be used as a base class for other `LightningModule`s that require metric tracking.

    The way the loss and metrics are logged is as follows:
        - The loss, defined by the `criterion` parameter, is logged as `"train/loss"`, `"val/loss"`, and `"test/loss"`,
          in the respective loops. This is useful to expose the loss to callbacks and optimizers that might need to
          monitor it.
        - The metrics, defined by the `metrics` parameter, are logged as `"train/<metric>"`, `"val/<metric>"`, and
          `"test/<metric>"`, in the respective loops. This is useful to expose the metrics to callbacks and optimizers
          that might need to monitor them.
        - For the "train" and "val" loss and metrics, the best values, either min or max depending on the metric, are
          also logged under `"<loop>/<metric>/best"`. This is useful to provide a known log entry from which to retrieve
          the best values across the whole run, for example to monitor them for automatic hyperparameter tuning.
    """

    task_level: Literal["node", "graph"]
    """The type of task the model is designed for, used to generate an example input batch."""

    def __init__(
        self,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion_target_dtype: str | torch.dtype | None = None,
        metrics: MetricCollection | None = None,
        num_node_features: int = None,
        num_edge_features: int = None,
        *args,
        **kwargs,
    ):
        """Initializes a `MetricTrackingLitModule`.

        Args:
            criterion: The loss function to use for training.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            criterion_target_dtype: Dtype to cast the targets to before passing them to the criterion.
                This can be useful for criteria that expect specific target types not typically provided by datasets,
                e.g. `BCEWithLogitsLoss` which expects float targets while class labels are usually provided as long.
            metrics: A collection of metrics to use for evaluation.
            num_node_features: The number of features per node in the input graph(s). If provided, it is used to
                generate an example input batch, useful for inspecting the model's input/output shapes.
            num_edge_features: The number of features per edge in the input graph(s). If provided, it is used to
                generate an example input batch, useful for inspecting the model's input/output shapes.
            *args: Additional positional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.criterion = criterion
        self._criterion_target_dtype = criterion_target_dtype
        if isinstance(self._criterion_target_dtype, str):
            self._criterion_target_dtype = import_from_module(self._criterion_target_dtype)
            if not isinstance(self._criterion_target_dtype, torch.dtype):
                raise ValueError(f"Invalid torch dtype string: {criterion_target_dtype}")
        # Use metric trackers to aggregate and keep track of min loss across epochs
        # this is useful for callbacks/optimizers that might want to monitor the loss
        self.train_loss_tracker = MetricTracker(MeanMetric(), maximize=False)
        self.val_loss_tracker = MetricTracker(MeanMetric(), maximize=False)
        # No tracker for test loss, since it should only be computed for one epoch
        self.test_loss = MeanMetric()

        self._base_metrics = metrics
        if self._base_metrics:
            # Use metric collection to group metrics in a single object, to update and log them together
            # torchmetrics recommends to use different instances of the metrics for train, val, and test
            # to avoid conflicts since the metrics are stateful
            # Just as for the loss, we wrap the metrics inside a tracker to help track the best values across epochs
            # Here, we explicitly set `maximize=None` to infer the best value from the underlying metric
            self.train_metrics_tracker = MetricTracker(metrics.clone(prefix="train/"), maximize=None)
            self.val_metrics_tracker = MetricTracker(metrics.clone(prefix="val/"), maximize=None)
            # No tracker for test metrics, since they should only be computed for one epoch
            self.test_metrics = metrics.clone(prefix="test/")

        available_data_hparams = [param is not None for param in [num_node_features, num_edge_features]]
        if any(available_data_hparams):
            if not all(available_data_hparams):
                log.warning(
                    "You provided the following hparams to generate an example input batch: "
                    f"{num_node_features=}, {num_edge_features=}. "
                    "No example batch will be generated because some hparams are missing."
                    "To suppress this warning, either provide missing hparams or set all of them to `None` to disable "
                    "example batch generation."
                )
            else:
                fake_dataset = FakeDataset(
                    num_graphs=2 if self.task_level == "graph" else 1,
                    num_channels=num_node_features,
                    edge_dim=num_edge_features,
                )
                self.example_input_array = Batch.from_data_list([data for data in fake_dataset])

    def save_hyperparameters(  # noqa: D102
        self,
        *args: Any,
        ignore: Sequence[str] | str | None = None,
        frame: types.FrameType | None = None,
        logger: bool = True,
    ) -> None:
        # add the `criterion` and `metrics` parameters to the list of ignored parameters
        if ignore is None:
            ignore = []
        ignore.extend(["criterion", "metrics"])

        # get the frame of the caller,
        # so that `super().save_hyperparameters()` can access the final caller's local variables
        current_frame = inspect.currentframe().f_back if frame is None else frame

        super().save_hyperparameters(*args, ignore=ignore, frame=current_frame, logger=logger)

    def model_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.

        Returns:
            A pair of tensors containing the loss and the (unnormalized) predictions (i.e. logits), respectively.
        """
        logits = self.forward(batch)
        target = batch.y
        if self._criterion_target_dtype is not None:
            target = target.to(self._criterion_target_dtype)
        loss = self.criterion(logits, target)
        return loss, logits

    def on_train_epoch_start(self) -> None:  # noqa: D102
        # Initialize new instances of the tracked loss/metrics for the new epoch
        self.train_loss_tracker.increment()
        if self._base_metrics:
            self.train_metrics_tracker.increment()

    def training_step(self, batch: Batch) -> torch.Tensor:  # noqa: D102
        # Perform the forward pass on the model and compute the loss
        loss, logits = self.model_step(batch)

        # Update the stateful loss and metrics
        self.train_loss_tracker.update(loss)
        if self._base_metrics:
            self.train_metrics_tracker.update(logits, batch.y)

        return loss

    def on_train_epoch_end(self) -> None:  # noqa: D102
        # Log the loss and metrics accumulated over the epoch, and the best values so far
        self.log("train/loss", self.train_loss_tracker.compute(), prog_bar=True)
        self.log("train/loss/best", self.train_loss_tracker.best_metric(), prog_bar=True)
        if self._base_metrics:
            self.log_dict(self.train_metrics_tracker.compute(), prog_bar=True)
            self.log_dict(pad_keys(self.train_metrics_tracker.best_metric(), postfix="/best"), prog_bar=True)

    def on_validation_epoch_start(self) -> None:  # noqa: D102
        # Initialize new instances of the tracked loss/metrics for the new epoch
        # Since by default Lightning executes validation step sanity checks before training starts,
        # this also makes sure that loss/metrics logged during the sanity check (i.e. 1st val increment)
        # are not used to compute loss/metrics in the 1st actual validation epoch (i.e. 2nd val increment)
        # This is a workaround to ignore sanity checks values, since trackers do not support deleting previous metrics,
        # that is simpler than the alternative of reinitializing the val trackers in `on_train_start`
        self.val_loss_tracker.increment()
        if self._base_metrics:
            self.val_metrics_tracker.increment()

    def validation_step(self, batch: Batch) -> None:  # noqa: D102
        # Perform the forward pass on the model and compute the loss
        loss, logits = self.model_step(batch)

        # Update the stateful loss and metrics
        self.val_loss_tracker.update(loss)
        if self._base_metrics:
            self.val_metrics_tracker.update(logits, batch.y)

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        # Log the loss and metrics accumulated over the epoch, and the best values so far
        self.log("val/loss", self.val_loss_tracker.compute(), prog_bar=True)
        self.log("val/loss/best", self.val_loss_tracker.best_metric(), prog_bar=True)
        if self._base_metrics:
            self.log_dict(self.val_metrics_tracker.compute(), prog_bar=True)
            self.log_dict(pad_keys(self.val_metrics_tracker.best_metric(), postfix="/best"), prog_bar=True)

    def test_step(self, batch: Batch) -> None:  # noqa: D102
        # Perform the forward pass on the model and compute the loss
        loss, logits = self.model_step(batch)

        # Update the stateful loss and log it, Lightning will take care to aggregate it over the epoch
        self.test_loss.update(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self._base_metrics:
            # Update the stateful metrics and log them, Lightning will take care to aggregate them over the epoch
            self.test_metrics.update(logits, batch.y)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        References:
            - PyTorch Lightning documentation on configuring optimizers:
              https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
