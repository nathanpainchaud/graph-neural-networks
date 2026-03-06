import copy
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

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

try:
    from ogb.graphproppred import PygGraphPropPredDataset

    no_ogb = False
except ImportError:
    PygGraphPropPredDataset = object
    no_ogb = True

log = RankedLogger(__name__, rank_zero_only=True)


class LightningDataset(LightningDataModule, ABC):
    """A `LightningDataModule` that wraps a PyG `Dataset` for use in typical train/val/test pipelines.

    This is a thin wrapper around a PyG `Dataset` that allows for easy integration with PyTorch Lightning. We avoid
    using PyG's own `LightningDataset` because it expects already instantiated datasets, and with the way PyG datasets
    are designed, this would mean the data would have been downloaded already. However, the philosophy of
    `LightningDataModule`s is to only download the data in the `prepare_data` hook.

    Therefore, we use a callable that returns the dataset to only instantiate the dataset in the `prepare_data` hook.
    """

    def __init__(self, dataset: Callable[..., Dataset], has_val: bool, has_test: bool, **kwargs) -> None:
        """Initializes a `LightningDataset`.

        Args:
            dataset: A callable that returns the dataset. See the class docstring for why this must be a callable.
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

        # Properties to cache the datasets after they have been created once in `setup`,
        # to avoid re-creating them on each call
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
        """Set up the train, val, test or predict datasets according to the stage.

        Args:
            stage: The stage of the trainer to set up the datasets for.
        """
        # On accessing each dataset, check if they have already been created and cached,
        # and avoid re-creating them if so
        if stage in (TrainerFn.FITTING, TrainerFn.PREDICTING):
            if self.train_dataset is None:
                self.train_dataset = self.get_dataset_split(TRAIN_SET)
            if self.val_dataset is None and self.val_dataloader is not None:
                self.val_dataset = self.get_dataset_split(VAL_SET)
        if stage in (TrainerFn.TESTING, TrainerFn.PREDICTING):  # noqa: SIM102
            if self.test_dataset is None and self.test_dataloader is not None:
                self.test_dataset = self.get_dataset_split(TEST_SET)

    @abstractmethod
    def get_dataset_split(self, split: str) -> Dataset:
        """Get a subset over a requested split of the dataset.

        Args:
            split: The split (e.g. 'train') of the dataset to return.

        Returns:
            Subset over the dataset's requested split.
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


class PreSplitLightningDataset(LightningDataset):
    """A `LightningDataset` for datasets with pre-defined splits, specified by a `split` init argument.

    This datamodule supports datasets that use the same API as PyG's built-in `ZINC` dataset, for example.
    """

    def __init__(self, dataset: Callable[[str], Dataset], *args, **kwargs) -> None:
        """Initializes a `PreSplitLightningDataset`.

        Args:
            dataset: A callable that takes a `split` argument (e.g. 'train') and returns the corresponding dataset
                split.
            *args: Additional positional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(dataset, *args, **kwargs)

    def get_dataset_split(self, split: str) -> Dataset:  # noqa: D102
        return self._dataset_init(split=split)


class OGBLightningDataset(LightningDataset):
    """A thin wrapper around the `ogb` datasets to use them in PyTorch Lightning."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        if no_ogb:
            raise ModuleNotFoundError(
                "No module named 'ogb' found. "
                "Install the project with the 'ogb' extra to use the 'OGBLightningDataset'."
            )

        # The following imports and safe globals additions are a workaround to load OGB datasets in torch>=2.6
        # until this issue is resolved: https://github.com/snap-stanford/ogb/issues/497
        # TODO: Remove this workaround once the issue linked above is resolved and the fix is released
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr  # noqa: PLC0415
        from torch_geometric.data.storage import GlobalStorage  # noqa: PLC0415

        torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

        super().__init__(*args, has_val=True, has_test=True, **kwargs)

        # Cache full dataset and generated splits to avoid loading/recomputing them on each call to `setup`
        self._dataset = None
        self._splits = None

    def get_dataset_split(self, split: str) -> Dataset:
        """Get a predefined OGB split (i.e. train, val, or test) for the dataset.

        Args:
            split: The split (e.g. 'train') of the dataset to return.

        Returns:
            Subset over the dataset's requested split.
        """
        if self._dataset is None:
            # Load dataset + generated splits only on 1st call and cache them
            # This 2nd call to `self._dataset_init()`, after the 1st call in `prepare_data()`, avoids downloading
            # the data again, since PyG datasets cache their data under their root directory
            self._dataset = cast(PygGraphPropPredDataset, self._dataset_init())
            ogb_splits = self._dataset.get_idx_split()
            # Map OGB split keys to our standard split keys
            # OGB uses 'valid' instead of 'val', but otherwise split keys are the same
            self._splits = {
                TRAIN_SET: ogb_splits["train"],
                VAL_SET: ogb_splits["valid"],
                TEST_SET: ogb_splits["test"],
            }

        return self._dataset[self._splits[split]]


class SplitLightningDataset(LightningDataset):
    """A `LightningDataset` that manages the splitting of the dataset according to a user-defined split function."""

    SplitFunction = Callable[[Dataset, torch.Tensor | None], DatasetSplit]

    def __init__(
        self,
        *args,
        split: SplitFunction,
        stratify: bool = True,
        split_idx: int = 0,
        on_conflict: Literal["raise", "warn", "ignore"] = "raise",
        **kwargs,
    ) -> None:
        """Initializes a `SplitLightningDataset`.

        Args:
            *args: Additional positional arguments to pass to the superclass.
            split: The callable to use for splitting the dataset.
            stratify: Whether to split the data in a stratified fashion, using the dataset's labels.
            split_idx: The split to use, in case of multiple splits, e.g. for cross-validation. If you have only one
                split, e.g. a typical train/val/test split, use the default value of 0 to access the only split.
            on_conflict: How to handle conflicts between the generated splits and saved splits:
                - "raise": Raise an error if the splits do not match.
                - "warn": Log a warning if the splits do not match, and use the saved splits.
                - "ignore": Silently ignore any conflicts and use the saved splits. Should only be used if you
                absolutely know what you are doing.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)
        self._split_fn = split
        self._stratify = stratify
        self._split_idx = split_idx
        self._on_conflict = on_conflict

        # Cache full dataset and generated splits to avoid loading/recomputing them on each call to `setup`
        self._dataset = None
        self._splits = None

    def get_dataset_split(self, split: str) -> Dataset:
        """Generates train, val, and test splits for the dataset, saves/compares them to a file, and returns one split.

        Args:
            split: The split (e.g. 'train') of the dataset to return.

        Returns:
            Subset over the dataset's requested split.
        """
        if self._dataset is None:
            # Load dataset + generated splits only on 1st call and cache them
            # This 2nd call to `self._dataset_init()`, after the 1st call in `prepare_data()`, avoids downloading
            # the data again, since PyG datasets cache their data under their root directory
            self._dataset = self._dataset_init()
            self._splits = self._get_splits(self._dataset)

        return self._dataset[self._splits[split]]

    def _get_splits(self, dataset: Dataset) -> dict[str, list[int]]:
        """Generates train, val, and test splits for the dataset, saves/compares them to a file, and returns one split.

        Args:
            dataset: The dataset to split.

        Returns:
            The indices of samples by subsets (e.g. 'train').
        """
        # Serialize the split function and its parameters to a string to use it as a unique identifier for the splits
        splits_repr = serialize_split_fn(self._split_fn, self._stratify)

        # Acquire a lock on the (possibly not existing) splits file
        # This is done to prevent multiple processes from overwriting each other's splits, in case multiple experiments
        # are launched at the same time that all require the same non-existing splits
        # Only use `dataset.root` as last resort to get the root, since some datasets, e.g. `TUDataset`, make it point
        # to the dataset's parent directory, one level up from the intuitive root
        dataset_root = Path(dataset.raw_dir).parent if hasattr(dataset, "raw_dir") else dataset.root
        splits_file = dataset_root / "splits" / f"{splits_repr}.json"

        with FileLock(str(splits_file.with_suffix(".lock"))):
            if not (splits_exist := splits_file.exists()):
                log.info(f"No saved splits match the requested splits. Saving new dataset splits in '{splits_file}'!")

                # Generate the splits for the dataset
                splits = self._split_fn(dataset, dataset.y if self._stratify else None)

                with splits_file.open("w") as f:
                    json.dump(splits, f, indent=4, sort_keys=True)

        # The lock on the splits file is automatically released after the context manager exits
        # once the splits have been written to the file

        if splits_exist:
            with splits_file.open() as f:
                previous_splits = json.load(f)

            if self._on_conflict == "ignore":
                log.info(
                    f"Found saved splits that match the requested splits. Using previously saved splits from "
                    f"'{splits_file}'!"
                )
                splits = previous_splits

            else:
                log.info(
                    f"Found saved splits that match the requested splits. Checking that the generated splits match "
                    f"saved splits from '{splits_file}'!"
                )

                # Generate the splits for the dataset
                splits = self._split_fn(dataset, dataset.y if self._stratify else None)

                if splits != previous_splits:
                    msg = (
                        f"Newly generated requested splits do not match the saved splits from '{splits_file}'. This "
                        f"might be due to changes in the dataset or split function, or a non-deterministic split "
                        f"function. If you are sure that the new splits are correct, delete the old file to save the "
                        f"new splits and get rid of this error."
                    )
                    if self._on_conflict == "raise":
                        raise RuntimeError(msg)
                    if self._on_conflict == "warn":
                        log.warning(msg)
                else:
                    log.info(f"Newly generated requested splits match the saved splits from '{splits_file}'!")

        return splits[self._split_idx]
