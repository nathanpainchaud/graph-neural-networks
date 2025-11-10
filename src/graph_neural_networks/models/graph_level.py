from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import aggr

from graph_neural_networks.models import GraphLitModule


class GraphLevelLitModule(GraphLitModule):
    """A LightningModule for GNNs aimed at graph-level tasks."""

    task_level = "graph"

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel", "regression"],
        encoder: nn.Module,
        readout: aggr.Aggregation,
        head: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        """Initializes a `GraphLevelLitModule`.

        Args:
            task: Prediction task for which to configure the GNN model.
            encoder: The GNN model used to encode the graph.
            readout: The readout operation to use to aggregate node features into a single graph-level representation.
            head: The prediction head used to make predictions based on the graph-level representation.
            transforms: Transformations with learnable parameters (e.g. embedding) to apply to the input graphs before
                passing them to the encoder.
            *args: Additional positional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["encoder", "readout", "head"])

        self.encoder = encoder
        self.readout = readout
        self.head = head

    def forward(self, data: Batch) -> torch.Tensor:
        """Perform a forward pass through the model on a batch of graphs.

        Args:
            data: A batch of graphs, represented as one big (disconnected) graph.

        Returns:
            The predicted logits for the input graphs in the batch.
        """
        x = data.x

        encoder_forward_kwargs = {}
        if getattr(self.encoder, "supports_edge_attr", False):
            encoder_forward_kwargs["edge_attr"] = data.edge_attr.float() if data.edge_attr is not None else None
        if getattr(self.encoder, "supports_edge_weight", False):
            encoder_forward_kwargs["edge_weight"] = data.edge_weight.float() if data.edge_weight is not None else None
        if getattr(self.encoder, "supports_batchnorm", False):
            encoder_forward_kwargs["batch"] = data.batch
            encoder_forward_kwargs["batch_size"] = data.batch_size

        # Cast input features that must be floats to floats
        x = self.encoder(x.float(), data.edge_index, **encoder_forward_kwargs)
        # Pass the batch size to readout operation to avoid CPU communication/graph breaks
        x = self.readout(x, ptr=data.ptr, dim_size=data.batch_size)
        # After the readout step, each graph as been reduced to one vector representation,
        # i.e. each element in the batch comes from a different graph, so we have to update the batch vector
        x = self.head(x, batch=torch.arange(data.batch_size, device=x.device), batch_size=data.batch_size)
        if self.hparams.task == "binary":
            x = x.squeeze(-1)  # Flatten the last dim when only one value is predicted
        return x
