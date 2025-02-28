from collections.abc import Callable

import torch
from torch import nn
from torch_geometric.data import Batch

from graph_neural_networks.models import MetricTrackingLitModule


class GraphLevelLitModule(MetricTrackingLitModule):
    """A LightningModule for GNNs aimed at graph-level tasks."""

    task_level = "graph"

    def __init__(
        self,
        encoder: nn.Module,
        readout: Callable[[torch.Tensor, torch.Tensor | None, int | None], torch.Tensor],
        head: nn.Module,
        *args,
        **kwargs,
    ):
        """Initializes a `GraphLevelLitModule`.

        Args:
            encoder: The GNN model used to encode the graph.
            readout: The readout operation to use to aggregate node features into a single graph-level representation.
            head: The prediction head used to make predictions based on the graph-level representation.
            *args: Additional positional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["encoder", "head"])

        self.encoder = encoder
        self.head = head

    def forward(self, data: Batch) -> torch.Tensor:
        """Perform a forward pass through the model on a batch of graphs.

        Args:
            data: A batch of graphs, represented as one big (disconnected) graph.

        Returns:
            The predicted logits for the input graphs in the batch.
        """
        x, batch, batch_size = data.x, data.batch, data.batch_size
        # Cast input features that must be floats to floats
        x = self.encoder(
            x.float(),
            data.edge_index,
            edge_weight=data.edge_weight.float() if data.edge_weight is not None else None,
            edge_attr=data.edge_attr.float() if data.edge_attr is not None else None,
            batch=batch,
            batch_size=batch_size,
        )
        # Pass the batch size to readout operation to avoid CPU communication/graph breaks
        x = self.hparams.readout(x, batch, batch_size)
        return self.head(x, batch=batch, batch_size=batch_size)
