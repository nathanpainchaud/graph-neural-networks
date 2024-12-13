import json
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock
from torch_geometric.data import Dataset
from torch_geometric.data.lightning import LightningDataset

from graph_neural_networks.data.split import DatasetSplit
from graph_neural_networks.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SplitLightningDataset(LightningDataset):
    """A LightningDataset that splits a full dataset, e.g. in k-fold, for use in typical train/val/test pipelines."""

    def __init__(self, dataset: Dataset, split: Callable[[Dataset], DatasetSplit], fold: int = 0, **kwargs) -> None:
        """Initializes a `SplitLightningDataset`.

        Args:
            dataset: The dataset to split.
            split: The function to use for splitting the dataset.
            fold: The fold to use, in case of multiple splits, e.g. for cross-validation. If you only need one split,
                e.g. for a typical train/val/test split, use the default value of 0 to access the single split.
            **kwargs: Additional keyword arguments to pass to `torch_geometric.loader.DataLoader`.
        """
        fold_split = self.get_splits(split, dataset)[fold]
        train_idx, val_idx, test_idx = fold_split["train"], fold_split["val"], fold_split.get("test")

        train_dataset, val_dataset = dataset[train_idx], dataset[val_idx]
        test_dataset = dataset[test_idx] if test_idx else None

        super().__init__(
            train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, pred_dataset=dataset, **kwargs
        )

    @staticmethod
    def get_splits(split_fn: Callable[[Dataset], DatasetSplit], dataset: Dataset) -> DatasetSplit:
        """Get the splits for the dataset according to `split_fn`, either by loading saved splits or creating new ones.

        Args:
            split_fn: The function to use for splitting the dataset.
            dataset: The dataset to split.

        Returns:
            The splits for the dataset. A list (to support multiple splits/folds) of dictionaries, each containing the
            indices for the train, test, and (optional) val sets of one split of the dataset. If you only need one
            split, e.g. for a typical train/val/test split, the list will contain a single dictionary/split.
        """
        # Serialize the split function and its kwargs to a string to use it as a unique identifier for the splits
        # The kwargs are sorted to ensure that the same kwargs in a different order result in the same string ID
        splits_id = ",".join(f"{k}={v}" for k, v in sorted(split_fn.keywords.items()))
        splits = None

        # Acquire a lock on the (possibly not existing) splits file
        # This is done to prevent multiple processes from overwriting each other's splits, in case multiple experiments
        # are launched at the same time that all require the same non-existing splits
        splits_file = Path(dataset.processed_dir) / "splits" / f"{splits_id}.json"

        with FileLock(str(splits_file.with_suffix(".lock"))):
            if not splits_file.exists():
                # If requested splits do not match saved splits for the dataset, create new splits
                log.info(f"No saved splits match the requested splits. Creating new dataset splits in '{splits_file}'!")
                splits = split_fn(dataset)

                with splits_file.open("w") as f:
                    json.dump(splits, f, indent=4, sort_keys=True)

        # The lock on the splits file is automatically released after the context manager exits
        # once the splits have been written to the file

        if not splits:
            log.info(f"Requested splits match previous splits. Loading saved splits from '{splits_file}'!")
            # Otherwise, load splits from the saved splits file
            with splits_file.open() as f:
                splits = json.load(f)

        return splits
