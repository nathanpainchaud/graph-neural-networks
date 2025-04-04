import copy
import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import FixtureRequest
from _pytest.logging import LogCaptureFixture
from lightning.pytorch.trainer.states import TrainerFn
from torch_geometric.data import Dataset

from graph_neural_networks.data import LightningDataset, SplitLightningDataset, k_fold, subsets_split
from graph_neural_networks.data.datamodule import OGBLightningDataset

from ...helpers.run import RunIf  # noqa: TID252


class AbstractLightningDatasetTest(ABC):
    """Abstract base class for `LightningDataset` datamodule tests, implementing generic fixtures and tests."""

    @staticmethod
    @abstractmethod
    @pytest.fixture
    def dm(*args, **kwargs) -> LightningDataset:
        """A Pytest fixture for the `LightningDataset` to test."""

    @staticmethod
    @pytest.fixture(scope="module")
    def batch_size() -> int:
        """A Pytest fixture for the default batch size to be used for the dataloaders."""
        return 1

    @staticmethod
    @pytest.fixture(scope="module", params=[TrainerFn.FITTING, TrainerFn.TESTING])
    def stage(request: FixtureRequest) -> TrainerFn:
        """A Pytest fixture for the stage for which to set up the `LightningDataset`."""
        return request.param

    @staticmethod
    @pytest.mark.parametrize("batch_size", [32, 128])
    def test_dataset_full(tmp_path: Path, dm: LightningDataset, batch_size: int) -> None:
        """Tests that `LightningDataset` is working as expected on full datasets (i.e. w/ train/val/test sets).

        It verifies that the data is downloaded correctly, that the necessary attributes were created
        (e.g., the dataloader objects), and that dtypes and batch sizes correctly match.

        Args:
            tmp_path: The temporary data path.
            dm: The datamodule to test.
            batch_size: Batch size of the data to be loaded by the dataloader.
        """
        # Ensure instantiating the datamodule (in the fixture) did not download the data
        assert not any(tmp_path.iterdir())

        # Ensure that `prepare_data` downloads the data, without assigning the subsets!
        dm.prepare_data()
        assert any(tmp_path.iterdir())
        assert not dm.train_dataset
        assert not dm.val_dataset
        assert not dm.test_dataset
        assert not dm.pred_dataset

        # Ensure that `setup` assigns the subsets (for the respective stages) and that dataloaders can be created
        dm.setup(TrainerFn.FITTING)
        assert dm.train_dataset
        assert dm.train_dataloader()
        assert dm.val_dataset
        assert dm.val_dataloader()

        dm.setup(TrainerFn.TESTING)
        assert dm.test_dataset
        assert dm.test_dataloader()

        dm.setup(TrainerFn.PREDICTING)
        assert dm.pred_dataset
        assert dm.predict_dataloader()

        # Ensure that the number of items in the datasets match
        num_datapoints = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
        assert num_datapoints == len(dm.pred_dataset)

        # Ensure that the dataloaders have the correct batch size and dtypes
        batch = next(iter(dm.train_dataloader()))
        x, y = batch.x, batch.y
        assert x.shape[-1] == dm.train_dataset.num_features
        assert len(y) == batch_size
        assert x.dtype in (torch.float32, torch.int64)
        assert y.dtype == torch.int64


class TestSplitLightningDataset(AbstractLightningDatasetTest):
    """Tests for the `SplitLightningDataset` datamodule class."""

    @staticmethod
    @pytest.fixture
    def dm(
        dataset_fn: Callable[[], Dataset],
        split: tuple[SplitLightningDataset.SplitFunction, bool, bool],
        stratify: bool,
        batch_size: int,
    ) -> SplitLightningDataset:
        """A Pytest fixture for a `SplitLightningDataset` datamodule.

        Args:
            dataset_fn: The dataset function fixture to use.
            split: A split function for the dataset and booleans indicating whether the split has val/test sets.
            stratify: Whether to split the data in a stratified fashion, using the dataset's class labels.
            batch_size: Batch size used for the dataloaders.
        """
        split_fn, has_val, has_test = split
        return SplitLightningDataset(
            dataset_fn, has_val, has_test, split=split_fn, stratify=stratify, batch_size=batch_size
        )

    @staticmethod
    @pytest.fixture(params=["mutag_dataset_fn"])
    def dataset_fn(request: FixtureRequest) -> Callable[[], Dataset]:
        """A Pytest fixture for a function that returns a PyG `Dataset` instance.

        Args:
            request: The pytest request builtin fixture.

        Returns:
            A function that returns a dataset.
        """
        return request.getfixturevalue(request.param)

    @staticmethod
    @pytest.fixture(scope="class", params=[k_fold, subsets_split])
    def split(request: FixtureRequest) -> tuple[SplitLightningDataset.SplitFunction, bool, bool]:
        """A Pytest fixture for the default split function and booleans indicating the availability of val/test sets.

        Args:
            request: The pytest request builtin fixture.

        Returns:
            A split function for the dataset and booleans indicating whether the split has val/test sets.
        """
        return request.param, True, True

    @staticmethod
    @pytest.fixture(scope="class")
    def stratify() -> bool:
        """A Pytest fixture for the default stratification setting to be used for the dataset splits."""
        return False

    @staticmethod
    @pytest.mark.parametrize("split", [(functools.partial(subsets_split, val_size=None), False, True)])
    def test_no_val_set(dm: SplitLightningDataset) -> None:
        """Tests that `SplitLightningDataset` is working as expected when no validation set is available.

        It verifies that no dataloader is set for the missing subset, and that setting up the datamodule does create
        a dataset for the missing subset.

        Args:
            dm: The datamodule to test.
        """
        # Ensure that the instantiated datamodule doesn't have a test dataloader
        assert not dm.val_dataloader

        # Ensure that `val_dataset` is still `None` even after calling `setup` for the fitting stage
        dm.prepare_data()
        dm.setup(TrainerFn.VALIDATING)
        assert not dm.val_dataset

    @staticmethod
    @pytest.mark.parametrize(
        "split",
        [
            (functools.partial(k_fold, test_fold=False), True, False),
            (functools.partial(subsets_split, test_size=None), True, False),
        ],
    )
    def test_no_test_set(dm: SplitLightningDataset) -> None:
        """Tests that `SplitLightningDataset` is working as expected when no test set is available.

        It verifies that no dataloader is set for the missing subset, and that setting up the datamodule does create
        a dataset for the missing subset.

        Args:
            dm: The datamodule to test.
        """
        # Ensure that the instantiated datamodule doesn't have a test dataloader
        assert not dm.test_dataloader

        # Ensure that `test_dataset` is still `None` even after calling `setup` for the testing stage
        dm.prepare_data()
        dm.setup(TrainerFn.TESTING)
        assert not dm.test_dataset

    @staticmethod
    def test_no_saved_splits(caplog: LogCaptureFixture, dm: SplitLightningDataset, stage: TrainerFn) -> None:
        """Test that `SplitLightningDataset` behaves correctly when no previous splits are available."""
        dm.prepare_data()
        with caplog.at_level(logging.INFO):
            dm.setup(stage)
        assert "No saved splits match the requested splits. Saving new dataset splits in" in caplog.text

    @staticmethod
    def test_matching_previous_splits(caplog: LogCaptureFixture, dm: SplitLightningDataset, stage: TrainerFn) -> None:
        """Test that `SplitLightningDataset` correctly checks against previous splits when they are available."""
        dm.prepare_data()
        dm_copy = copy.deepcopy(dm)  # Copy dm before setting internal state w/ `setup` to simulate, e.g. another worker
        dm.setup(stage)
        with caplog.at_level(logging.INFO):
            dm_copy.setup(stage)
        assert (
            "Found saved splits that match the requested splits. Checking that the generated splits match saved splits "
            "from" in caplog.text
        )
        assert "Newly generated requested splits match the saved splits from" in caplog.text

    @staticmethod
    @pytest.mark.parametrize("split", [(functools.partial(k_fold, random_state=None), True, True)])
    def test_mismatching_previous_splits(
        caplog: LogCaptureFixture, dm: SplitLightningDataset, stage: TrainerFn
    ) -> None:
        """Test that `SplitLightningDataset` correctly errors on previous splits that do not match new splits.

        Notes:
            - The `split` function is parametrized to be non-deterministic so that splits are different between calls
        """
        dm.prepare_data()
        dm_copy = copy.deepcopy(dm)  # Copy dm before setting internal state w/ `setup` to simulate, e.g. another worker
        dm.setup(stage)
        with caplog.at_level(logging.INFO), pytest.raises(RuntimeError) as exc_info:
            dm_copy.setup(stage)

        assert (
            "Found saved splits that match the requested splits. Checking that the generated splits match saved splits "
            "from" in caplog.text
        )
        assert exc_info.type is RuntimeError
        assert "Newly generated requested splits do not match the saved splits from" in str(exc_info.value)


@RunIf(ogb=True)
@pytest.mark.slow
class TestOGBLightningDataset(AbstractLightningDatasetTest):
    """Tests for the `OGBLightningDataset` datamodule class."""

    @staticmethod
    @pytest.fixture
    def dm(dataset_fn: Callable[[], Dataset], batch_size: int) -> OGBLightningDataset:
        """A Pytest fixture for an `OGBLightningDataset` datamodule.

        Args:
            dataset_fn: The dataset function fixture to use.
            batch_size: Batch size used for the dataloaders.
        """
        return OGBLightningDataset(dataset_fn, batch_size=batch_size)

    @staticmethod
    @pytest.fixture(params=["ogbg_molhiv_dataset_fn"])
    def dataset_fn(request: FixtureRequest) -> Callable[[], Dataset]:
        """A Pytest fixture for a function that returns a PyG `Dataset` instance.

        Args:
            request: The pytest request builtin fixture.

        Returns:
            A function that returns a dataset.
        """
        return request.getfixturevalue(request.param)
