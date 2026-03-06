from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from graph_neural_networks.utils.logging_utils import (
    create_predictions_dataframe,
    log_dataframe,
)


class GraphLevelPredictionWriter(BasePredictionWriter):
    """Callback to log graph-level predictions to the configured logger/experiment tracker."""

    def __init__(
        self,
        output_dir: str,
        save_fit_predictions: bool,
        save_test_predictions: bool,
        filename_format: str = "{}_predictions.csv",
        output_labels: Sequence[str] | None = None,
        samplewise_op: Literal["softmax", "argmax"] | list[Literal["softmax", "argmax"]] | None = None,
    ) -> None:
        """Initializes a `GraphLevelPredictionWriter` instance.

        Args:
            output_dir: Directory where prediction files will be saved.
            save_fit_predictions: Whether the dataloaders passed to the prediction loop will include training and
                validation dataloaders.
            save_test_predictions: Whether the dataloaders passed to the prediction loop will include the test
                dataloader.
            filename_format: A format string for naming the output files. It should include a placeholder
                that will be replaced with the subset (e.g., "train", "val", "test") and samplewise operation applied.
            output_labels: Sequence of label names corresponding to prediction columns, for models that return multiple
                values per sample (e.g. class logits).
            samplewise_op: Operation(s) to apply to model outputs on a per-sample basis before saving/logging.
        """
        super().__init__(write_interval="epoch")

        self.output_dir = output_dir
        self.predictions_dataloaders = []
        if save_fit_predictions:
            self.predictions_dataloaders.extend(["train", "val"])
        if save_test_predictions:
            self.predictions_dataloaders.append("test")
        self.filename_format = filename_format
        self.output_labels = output_labels
        self.samplewise_ops = samplewise_op
        if samplewise_op is None or isinstance(samplewise_op, str):
            # If a single samplewise operation is provided, convert it to a list for consistency
            self.samplewise_ops = [samplewise_op]
        if None not in self.samplewise_ops:
            # If a samplewise operation is specified, also save unmodified predictions by adding None as an operation
            self.samplewise_ops.append(None)

    def _format_predictions(
        self, dataloader_preds: list[torch.Tensor], dataloader_batch_indices: list[list[int]]
    ) -> tuple[np.ndarray, list[int]]:
        """Format nested predictions and batch indices from a single dataloader to flat structures.

        Args:
            dataloader_preds: Predictions from a single dataloader (list of batch predictions).
            dataloader_batch_indices: Batch indices corresponding to the predictions.

        Returns:
            A tuple containing:
                - Predictions as a numpy array
                - Flattened list of batch indices

        Raises:
            ValueError: If the number of batch indices is not equal to the number of predictions.
        """
        # Convert predictions to numpy for DataFrame creation
        all_predictions = torch.cat(dataloader_preds).cpu().numpy()

        # Flatten batch indices
        all_batch_indices = []
        for batch_idx_list in dataloader_batch_indices:
            all_batch_indices.extend(batch_idx_list)

        # Validate that batch indices and predictions match
        if len(all_batch_indices) != len(all_predictions):
            raise ValueError(
                f"Number of batch indices ({len(all_batch_indices)}) does not match "
                f"number of predictions ({len(all_predictions)})"
            )

        return all_predictions, all_batch_indices

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: list[torch.Tensor | list[torch.Tensor]],
        batch_indices: list[list[list[int]]],
    ) -> None:
        """Logs predictions at the end of an epoch, saving each dataloader's predictions to a separate file.

        This method collects predictions from different dataloaders (train/val/test), saves them to separate
        CSV files, and logs them to the configured experiment tracker if available.

        Note:
            For WandbLogger, predictions are logged as interactive Tables.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
            predictions: A sequence of predictions from each dataloader. Each element is a list of batch predictions.
            batch_indices: A sequence of batch indices corresponding to the predictions.
        """
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(predictions[0], torch.Tensor):
            # If there is only one dataloader, wrap predictions in a list to match expected structure
            predictions = [predictions]

        # Iterate through each dataloader's predictions
        for subset, dataloader_preds, dataloader_batch_indices in zip(
            self.predictions_dataloaders, predictions, batch_indices, strict=True
        ):
            # Format predictions and batch indices
            dataloader_preds, dataloader_batch_indices = self._format_predictions(  # noqa: PLW2901
                dataloader_preds, dataloader_batch_indices
            )

            for samplewise_op in self.samplewise_ops:
                # Create DataFrame using utility function
                df = create_predictions_dataframe(
                    predictions=dataloader_preds,
                    batch_indices=dataloader_batch_indices,
                    output_labels=self.output_labels,
                    samplewise_op=samplewise_op,
                )

                # Save to CSV file
                op_suffix = f"_{samplewise_op}" if samplewise_op is not None else ""
                filepath = output_dir / self.filename_format.format(subset + op_suffix)
                df.to_csv(filepath, index=False)

                # Log to experiment tracker if available
                if trainer.logger is not None:
                    log_dataframe(df, trainer.logger, filepath.stem)
