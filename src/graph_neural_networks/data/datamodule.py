import copy
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from filelock import FileLock
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import IterableDataset
from torch_geometric.data import Dataset
from torch_geometric.data.lightning.datamodule import kwargs_repr
from torch_geometric.loader import DataLoader

from graph_neural_networks.data.split import TEST_SET, TRAIN_SET, VAL_SET, DatasetSplit, serialize_split_fn
from graph_neural_networks.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LightningDataset(LightningDataModule, ABC):
    """A `LightningDataModule` that wraps a PyG `Dataset` for use in typical train/val/test pipelines.

    This is a thin wrapper around a PyG `Dataset` that allows for easy integration with PyTorch Lightning. We avoid
    using PyG's own `LightningDataset` because it expects already instantiated datasets, and with the way PyG datasets
    are designed, this would mean the data would have been downloaded already. However, the philosophy of
    `LightningDataModule`s is to only download the data in the `prepare_data` hook.

    Therefore, we use a callable that returns the dataset to only instantiate the dataset in the `prepare_data` hook.
    """

    def __init__(self, dataset: Callable[[], Dataset], has_val: bool, has_test: bool, **kwargs) -> None:
        """Initializes a `LightningDataset`.

        Args:
            dataset: A callable that returns the dataset. See the class docstring for why this is a callable.
            has_val: Whether the dataset has a validation set.
            has_test: Whether the dataset has a test set.
            **kwargs: Additional keyword arguments to pass to `torch_geometric.loader.DataLoader`.
        """
        super().__init__()

        self._dataset_init = dataset
        self.kwargs = kwargs

        # Remove the val and test dataloaders if the dataset does not have sets for them
        if not has_val:
            self.val_dataloader = None
        if not has_test:
            self.test_dataloader = None

        if "shuffle" in kwargs:
            log.warning(
                f"The 'shuffle={kwargs['shuffle']}' option is ignored in '{self.__class__.__name__}'. Remove it from "
                f"the argument list to disable this warning"
            )
            del kwargs["shuffle"]

        self.kwargs = kwargs

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None

    def __repr__(self) -> str:  # noqa: D105
        kwargs = kwargs_repr(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            pred_dataset=self.pred_dataset,
            **self.kwargs,
        )
        return f"{self.__class__.__name__}({kwargs})"

    def prepare_data(self) -> None:
        """Instantiate the PyG dataset to download the data (if necessary)."""
        # Instantiate the PyG dataset (by calling `_dataset_init`) to trigger the download of the data.
        # Do not assign the resulting dataset to `self.*` to respect the design of LightningDataModule
        # where `prepare_data` should not assign state (since it is only called on the main process)
        self._dataset_init()

    def setup(self, stage: TrainerFn) -> None:
        """Set up the train, val, test or predict datasets according to the stage, splitting into subsets if needed.

        Args:
            stage: The stage of the trainer to set up the datasets for.
        """
        # Instantiate the PyG dataset again, this time to assign it. Avoids downloading the data again since PyG
        # datasets cache their data under their root directory
        dataset = self._dataset_init()

        if stage == TrainerFn.PREDICTING:
            self.pred_dataset = dataset

        else:
            # Split the dataset into train, val, and test sets
            train_idx, val_idx, test_idx = self.get_splits(dataset)

            if stage == TrainerFn.FITTING:
                self.train_dataset = dataset[train_idx]
                self.val_dataset = dataset[val_idx] if val_idx is not None else None
            elif stage == TrainerFn.TESTING:
                self.test_dataset = dataset[test_idx] if test_idx is not None else None

    @abstractmethod
    def get_splits(self, dataset: Dataset) -> tuple[list[int], list[int] | None, list[int] | None]:
        """Get the splits of the dataset between train, val, and test sets.

        Args:
            dataset: The dataset to split.

        Returns:
            The indices of samples for the train, val, and test sets.
        """

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= self.kwargs.get("sampler", None) is None
        shuffle &= self.kwargs.get("batch_sampler", None) is None

        return DataLoader(self.train_dataset, shuffle=shuffle, **self.kwargs)

    def _eval_dataloader(self, dataset: Dataset) -> DataLoader:
        assert dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)

        return DataLoader(dataset, shuffle=False, **kwargs)

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return self._eval_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        return self._eval_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:  # noqa: D102
        return self._eval_dataloader(self.pred_dataset)


class SplitLightningDataset(LightningDataset):
    """A `LightningDataset` that manages the splitting of the dataset according to a user-defined split function."""

    SplitFunction = Callable[[Dataset, torch.Tensor | None], DatasetSplit]

    def __init__(self, *args, split: SplitFunction, stratify: bool = False, split_idx: int = 0, **kwargs) -> None:
        """Initializes a `SplitLightningDataset`.

        Args:
            *args: Additional positional arguments to pass to the superclass.
            split: The callable to use for splitting the dataset.
            stratify: Whether to split the data in a stratified fashion, using the dataset's class labels.
            split_idx: The split to use, in case of multiple splits, e.g. for cross-validation. If you have only one
                split, e.g. a typical train/val/test split, use the default value of 0 to access the only split.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)
        self._split_fn = split
        self._stratify = stratify
        self._split_idx = split_idx

    def get_splits(self, dataset: Dataset) -> tuple[list[int], list[int] | None, list[int] | None]:
        """Generates train, val, and test splits for the dataset, saves/compares them to a file, and returns one split.

        Args:
            dataset: The dataset to split.

        Returns:
            The indices of samples for the train, val, and test sets.
        """
        # Generate the splits for the dataset
        splits = self._split_fn(dataset, dataset.y if self._stratify else None)

        # Serialize the split function and its parameters to a string to use it as a unique identifier for the splits
        splits_repr = serialize_split_fn(self._split_fn, self._stratify)

        # Acquire a lock on the (possibly not existing) splits file
        # This is done to prevent multiple processes from overwriting each other's splits, in case multiple experiments
        # are launched at the same time that all require the same non-existing splits
        splits_file = Path(dataset.processed_dir, "splits", f"{splits_repr}.json")

        with FileLock(str(splits_file.with_suffix(".lock"))):
            if save_splits := not splits_file.exists():
                log.info(f"No saved splits match the requested splits. Saving new dataset splits in '{splits_file}'!")
                with splits_file.open("w") as f:
                    json.dump(splits, f, indent=4, sort_keys=True)

        # The lock on the splits file is automatically released after the context manager exits
        # once the splits have been written to the file

        if not save_splits:
            log.info(
                f"Requested splits match previous splits. Asserting that generated splits match previous splits from "
                f"'{splits_file}'!"
            )
            with splits_file.open() as f:
                previous_splits = json.load(f)

            if splits != previous_splits:
                raise RuntimeError(
                    f"The splits generated by '{splits_repr}' do not match the splits saved in '{splits_file}'. "
                    "This might be due to changes in the dataset or the split function. If you are sure that the new "
                    "splits are correct, delete the old file to regenerate the splits and get rid of this error."
                )

        split = splits[self._split_idx]
        return split[TRAIN_SET], split.get(VAL_SET), split.get(TEST_SET)
