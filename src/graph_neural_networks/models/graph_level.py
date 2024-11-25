from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch

from graph_neural_networks.utils.logging_utils import prefix


class GraphLevelLitModule(LightningModule):
    """A LightningModule for training GNNs aimed at graph-level tasks."""

    def __init__(
        self,
        encoder: nn.Module,
        graph_pooling: Callable[[torch.Tensor, torch.Tensor | None, int | None], torch.Tensor],
        head: nn.Module,
        criterion: nn.Module,
        metrics: dict[str, nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        """Initializes a `GraphLevelLitModule`.

        :param encoder: The GNN model used to encode the graph.
        :param graph_pooling: The graph pooling operation to use to aggregate node features into a single graph-level
            representation.
        :param head: The prediction head used to make predictions based on the graph-level representation.
        :param criterion: The loss function to use for training.
        :param metrics: A mapping of metric names to metric functions to use for evaluation.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["encoder", "head", "criterion", "metrics"])

        self.encoder = encoder
        self.head = head
        self.criterion = criterion
        self.metrics = metrics

    def forward(self, data: Batch) -> torch.Tensor:
        """Perform a forward pass through the model on a batch of graphs.

        :param data: A batch of graphs, represented as one big (disconnected) graph.
        :return: The predicted logits for the input graphs in the batch.
        """
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, data.batch_size
        x = self.encoder(x, edge_index, batch=batch, batch_size=batch_size)
        # Pass the batch size to graph pooling operation to avoid CPU communication/graph breaks
        x = self.hparams.graph_pooling(x, batch, batch_size)
        return self.head(x, batch=batch, batch_size=batch_size)

    def model_step(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :return: A tuple containing (in order):
            - A tensor of (unnormalized) predictions.
            - A mapping of metric names to metric values, including a required 'loss' entry.
        """
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y)
        metrics = {"loss": loss}
        metrics.update({name: metric(logits, batch.y) for name, metric in self.metrics.items()})
        return logits, metrics

    def training_step(self, batch: Batch) -> torch.Tensor:  # noqa: D102
        logits, metrics = self.model_step(batch)
        self.log_dict(
            prefix(metrics, "train/"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.batch_size
        )
        return metrics["loss"]

    def validation_step(self, batch: Batch) -> None:  # noqa: D102
        # TODO: Log "optimized_metric" so that Optuna can access it for hyperparameter optimization
        logits, metrics = self.model_step(batch)
        self.log_dict(prefix(metrics, "val/"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.batch_size)

    def test_step(self, batch: Batch) -> None:  # noqa: D102
        logits, metrics = self.model_step(batch)
        self.log_dict(
            prefix(metrics, "test/"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.batch_size
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
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
