import copy
import json
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import IterableDataset
from torch_geometric.data import Dataset
from torch_geometric.data.lightning.datamodule import kwargs_repr
from torch_geometric.loader import DataLoader

from graph_neural_networks.data.split import TEST_SET, TRAIN_SET, VAL_SET, DatasetSplit
from graph_neural_networks.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SplitLightningDataset(LightningDataModule):
    """A `LightningDataModule` that splits a full dataset, e.g. in k-fold, for use in typical train/val/test pipelines.

    We avoid inheriting from PyG's `LightningDataset` because it expects already instantiated datasets, and with the way
    PyG datasets are designed, this would mean the data would have been downloaded already. However, the philosophy of
    `LightningDataModule`s is to only download the data in the `prepare_data` hook.

    Therefore, we use a callable that returns the dataset to only instantiate the dataset in the `prepare_data` hook.
    """

    def __init__(
        self,
        dataset: Callable[[], Dataset],
        split: Callable[[Dataset], DatasetSplit],
        has_val: bool,
        has_test: bool,
        split_idx: int = 0,
        **kwargs,
    ) -> None:
        """Initializes a `SplitLightningDataset`.

        Args:
            dataset: A callable that returns the dataset to split. See the class docstring for why this is a callable.
            split: The callable to use for splitting the dataset. Note that `repr` is called on it to serialize it to a
                string to save/load splits. So you might want to implement a custom `__repr__` wrapper for it to make
                sure only the relevant parameters are serialized.
            has_val: Whether the split will include a validation set.
            has_test: Whether the split will include a test set.
            split_idx: The split to use, in case of multiple splits, e.g. for cross-validation. If you have only one
                split, e.g. a typical train/val/test split, use the default value of 0 to access the only split.
            **kwargs: Additional keyword arguments to pass to `torch_geometric.loader.DataLoader`.
        """
        super().__init__()

        self._dataset_init = dataset
        self._split_fn = split
        self.split_idx = split_idx

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
            split = self.get_splits(self._split_fn, dataset)[self.split_idx]
            train_idx, val_idx, test_idx = split[TRAIN_SET], split.get(VAL_SET), split.get(TEST_SET)

            if stage == TrainerFn.FITTING:
                self.train_dataset = dataset[train_idx]
                self.val_dataset = dataset[val_idx] if val_idx else None
            elif stage == TrainerFn.TESTING:
                self.test_dataset = dataset[test_idx] if test_idx else None

    @staticmethod
    def get_splits(split_fn: Callable[[Dataset], DatasetSplit], dataset: Dataset) -> DatasetSplit:
        """Get the splits for the dataset according to `split_fn`, either by loading saved splits or creating new ones.

        Args:
            split_fn: The function to use for splitting the dataset.
            dataset: The dataset to split.

        Returns:
            The splits for the dataset. A list of dictionaries (to support multiple splits), each containing the
            indices for the train, test, and (optional) val sets of one split of the dataset. If you only need one
            split, e.g. for a typical train/val/test split, the list will contain a single dictionary/split.
        """
        # Generate the splits for the dataset
        splits = split_fn(dataset)

        # Serialize the split function and its parameters to a string to use it as a unique identifier for the splits
        splits_repr = repr(split_fn)

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

        return splits

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
