import functools
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from _pytest.logging import LogCaptureFixture

from graph_neural_networks.data import k_fold, subsets_split
from graph_neural_networks.data.split import TEST_SET, TRAIN_SET, VAL_SET, DatasetSplit, serialize_split_fn


@pytest.fixture(scope="module", params=[100])
def data(request: FixtureRequest) -> np.ndarray:
    """A Pytest fixture of random data simulating a dataset's features."""
    return np.random.default_rng().random((request.param, 2))


@pytest.fixture(scope="module", params=[None, "classification", "regression"])
def labels(request: FixtureRequest, data: Sequence[Any]) -> np.ndarray | None:
    """A Pytest fixture of random labels simulating a dataset's supervised labels.

    Args:
        request: The pytest request builtin fixture.
        data: The random features for which to generate matching labels.
    """
    rng = np.random.default_rng()
    match request.param:
        case "classification":
            return rng.integers(low=2, size=len(data))
        case "regression":
            return rng.random(size=len(data))
        case _:
            return None


class AbstractSplitTest(ABC):
    """Abstract base class for split function tests, implementing generic fixtures and tests."""

    @staticmethod
    @abstractmethod
    @pytest.fixture
    def splits_fn(*args, **kwargs) -> Callable[[], DatasetSplit]:
        """A Pytest fixture that returns a split function, to be able to call it multiple times for reproducibility."""

    @staticmethod
    @pytest.fixture
    def random_state() -> int:
        """A Pytest fixture for the random state to use when generating the splits."""
        return 12345

    @staticmethod
    @pytest.mark.parametrize("random_state", [42, 12345])
    def test_reproducible_splits_with_seed(splits_fn: Callable[[], DatasetSplit]) -> None:
        """Test that the splits are reproducible when the random state is set."""
        assert splits_fn() == splits_fn()

    @staticmethod
    def test_random_splits_without_seed(splits_fn: Callable[[], DatasetSplit]) -> None:
        """Test that the splits are different when the random state is not set."""
        assert splits_fn() != splits_fn(random_state=None)
        assert splits_fn(random_state=None) != splits_fn(random_state=None)


class TestKFold(AbstractSplitTest):
    """Tests for the k-fold split function."""

    @staticmethod
    @pytest.fixture
    def splits_fn(
        data: np.ndarray,
        labels: np.ndarray | None,
        n_splits: int,
        test_fold: bool,
        holdout_test_size: float | int | None,
        random_state: int | np.random.RandomState | None,
    ) -> Callable[[], DatasetSplit]:
        """A Pytest fixture for the k-fold split function.

        Args:
            data: The random features corresponding to the dataset to split.
            labels: The random labels associated with the features, if any.
            n_splits: The number of splits to generate.
            test_fold: Whether to reserve a fold for testing in each of the splits.
            holdout_test_size: The size of the holdout test set to split from the training set before creating the K
                folds.
            random_state: The random state to use for reproducibility.
        """
        return functools.partial(
            k_fold,
            data,
            labels,
            n_splits=n_splits,
            test_fold=test_fold,
            holdout_test_size=holdout_test_size,
            random_state=random_state,
        )

    @staticmethod
    @pytest.fixture(params=[5, 10])
    def n_splits(request: FixtureRequest) -> int:
        """A Pytest fixture for the number of splits to generate.

        Args:
            request: The pytest request builtin fixture.
        """
        return request.param

    @staticmethod
    @pytest.fixture
    def test_fold() -> bool:
        """A Pytest fixture for whether to reserve a fold for testing in each split."""
        return True

    @staticmethod
    @pytest.fixture
    def holdout_test_size() -> int | float | None:
        """A Pytest fixture for the holdout test size to use when generating the splits."""
        return None

    @staticmethod
    @pytest.mark.parametrize("test_fold", [False])
    @pytest.mark.parametrize("holdout_test_size", [None, 0])
    def test_train_val_only(splits_fn: Callable[[], DatasetSplit], data: np.ndarray, n_splits: int) -> None:
        """Test splits with only train and val sets."""
        splits = splits_fn()
        # Ensure that the correct number of splits are generated
        assert len(splits) == n_splits
        # Ensure that all splits only have train and val sets
        assert all({TRAIN_SET, VAL_SET} == split.keys() for split in splits)
        # Ensure that all val folds are disjoint
        assert all(
            set(split1[VAL_SET]).isdisjoint(split2[VAL_SET]) for split1, split2 in itertools.combinations(splits, 2)
        )
        # Ensure that in all splits, the train and val sets are disjoint
        assert all(set(split[TRAIN_SET]).isdisjoint(split[VAL_SET]) for split in splits)
        # Ensure that the cardinality of the val sets is correct
        val_size = len(data) // n_splits
        assert all(len(set(split[VAL_SET])) == val_size for split in splits)
        # Ensure that in all splits, the train and val sets cover the full dataset
        assert all(len(set(split[TRAIN_SET]) | set(split[VAL_SET])) == len(data) for split in splits)

    @staticmethod
    @pytest.mark.parametrize("test_fold", [True])
    @pytest.mark.parametrize("holdout_test_size", [None, 0])
    def test_test_fold(splits_fn: Callable[[], DatasetSplit], data: np.ndarray, n_splits: int) -> None:
        """Test splits with a different fold reserved for testing in each split."""
        splits = splits_fn()
        # Ensure that the correct number of splits are generated
        assert len(splits) == n_splits
        # Ensure that all splits have train, val and test sets
        assert all({TRAIN_SET, VAL_SET, TEST_SET} == split.keys() for split in splits)
        # Ensure that all val and test folds are disjoint across splits
        for split1, split2 in itertools.combinations(splits, 2):
            assert set(split1[VAL_SET]).isdisjoint(split2[VAL_SET])
            assert set(split1[TEST_SET]).isdisjoint(split2[TEST_SET])
        # Ensure that in all splits, the train, val and test sets are disjoint
        for split in splits:
            assert set(split[TRAIN_SET]).isdisjoint(split[VAL_SET])
            assert set(split[TRAIN_SET]).isdisjoint(split[TEST_SET])
            assert set(split[VAL_SET]).isdisjoint(split[TEST_SET])
        # Ensure that the cardinalities of the val and test sets are correct
        test_card = val_card = len(data) // n_splits
        assert all(len(set(split[VAL_SET])) == val_card for split in splits)
        assert all(len(set(split[TEST_SET])) == test_card for split in splits)
        # Ensure that in all splits, the train, val and test sets cover the full dataset
        assert all(
            len(set(split[TRAIN_SET]) | set(split[VAL_SET]) | set(split[TEST_SET])) == len(data) for split in splits
        )

    @staticmethod
    @pytest.mark.parametrize("test_fold", [False])
    @pytest.mark.parametrize("holdout_test_size", [0.1, 20])
    def test_holdout_test_size(
        splits_fn: Callable[[], DatasetSplit], data: np.ndarray, n_splits: int, holdout_test_size: float | int
    ) -> None:
        """Test splits with a holdout test set of a specific size that is the same across all splits."""
        splits = splits_fn()
        # Ensure that the correct number of splits are generated
        assert len(splits) == n_splits
        # Ensure that all splits have train, val and test sets
        assert all({TRAIN_SET, VAL_SET, TEST_SET} == split.keys() for split in splits)
        # Ensure that the test set is the same across all splits
        assert all(set(split[TEST_SET]) == set(splits[0][TEST_SET]) for split in splits)
        # Ensure that all val folds are disjoint
        assert all(
            set(split1[VAL_SET]).isdisjoint(split2[VAL_SET]) for split1, split2 in itertools.combinations(splits, 2)
        )
        # Ensure that in all splits, the train, val and test sets are disjoint
        for split in splits:
            assert set(split[TRAIN_SET]).isdisjoint(split[VAL_SET])
            assert set(split[TRAIN_SET]).isdisjoint(split[TEST_SET])
            assert set(split[VAL_SET]).isdisjoint(split[TEST_SET])
        # Ensure that the cardinality of the test sets is correct
        # (only check this for the first split since we've already checked that all test sets are equal)
        test_card = holdout_test_size if isinstance(holdout_test_size, int) else int(holdout_test_size * len(data))
        assert len(set(splits[0][TEST_SET])) == test_card
        # Ensure that the cardinality of the val sets is correct
        val_card = (len(data) - test_card) // n_splits
        assert all(len(set(split[VAL_SET])) == val_card for split in splits)
        # Ensure that in all splits, the train, val and test sets cover the full dataset
        assert all(
            len(set(split[TRAIN_SET]) | set(split[VAL_SET]) | set(split[TEST_SET])) == len(data) for split in splits
        )

    @staticmethod
    @pytest.mark.parametrize("test_fold", [True])
    @pytest.mark.parametrize("holdout_test_size", [0.1, 20])
    def test_test_fold_and_holdout_test_size(splits_fn: Callable[[], DatasetSplit]) -> None:
        """Test that an error is raised when both `holdout_test_size` and `test_fold` are set."""
        with pytest.raises(ValueError, match="both `holdout_test_size` and `test_fold`, which are mutually exclusive"):
            splits_fn()


class TestSubsetsSplit(AbstractSplitTest):
    """Tests for the subsets split function."""

    @staticmethod
    @pytest.fixture
    def splits_fn(
        data: np.ndarray,
        labels: np.ndarray | None,
        val_size: float | int | None,
        test_size: float | int | None,
        random_state: int | np.random.RandomState | None,
    ) -> Callable[[], DatasetSplit]:
        """A Pytest fixture for the subsets split function.

        Args:
            data: The random features corresponding to the dataset to split.
            labels: The random labels associated with the features, if any.
            val_size: The size of the validation set to split.
            test_size: The size of the test set to split.
            random_state: The random state to use for reproducibility.
        """
        return functools.partial(
            subsets_split, data, labels, val_size=val_size, test_size=test_size, random_state=random_state
        )

    @staticmethod
    @pytest.fixture
    def val_size() -> float | int:
        """A Pytest fixture for the size of the validation set to split."""
        return 0.1

    @staticmethod
    @pytest.fixture
    def test_size() -> float | int:
        """A Pytest fixture for the size of the test set to split."""
        return 0.2

    @staticmethod
    def _get_split(splits_fn: Callable[[], DatasetSplit]) -> dict[str, list[int]]:
        splits = splits_fn()
        # Ensure that the correct number of splits are generated
        assert len(splits) == 1
        return splits[0]

    @staticmethod
    @pytest.mark.parametrize("val_size", [None, 0])
    @pytest.mark.parametrize("test_size", [None, 0])
    def test_train_only(caplog: LogCaptureFixture, splits_fn: Callable[[], DatasetSplit], data: np.ndarray) -> None:
        """Test split with only a train set."""
        # Ensure that the warning is logged when no val or test sets are present
        with caplog.at_level(logging.WARNING):
            split = TestSubsetsSplit._get_split(splits_fn)
        assert "No validation or test set configured. Returning the full dataset as train set." in caplog.text
        # Ensure that only a train set is present
        assert {TRAIN_SET} == split.keys()
        # Ensure that the train set covers the full dataset
        assert len(set(split[TRAIN_SET])) == len(data)

    @staticmethod
    def _test_train_other_set(
        split: dict[str, list[int]], other_set: str, other_size: float | int, data: np.ndarray
    ) -> None:
        # Ensure that only train and other sets are present
        assert {TRAIN_SET, other_set} == split.keys()
        # Ensure that the train and other sets are disjoint
        assert set(split[TRAIN_SET]).isdisjoint(split[other_set])
        # Ensure that the cardinality of the other set is correct
        other_card = other_size if isinstance(other_size, int) else int(other_size * len(data))
        assert len(set(split[other_set])) == other_card
        # Ensure that the train and val sets cover the full dataset
        assert len(set(split[TRAIN_SET]) | set(split[other_set])) == len(data)

    @staticmethod
    @pytest.mark.parametrize("val_size", [0.2, 10])
    @pytest.mark.parametrize("test_size", [None, 0])
    def test_train_val(splits_fn: Callable[[], DatasetSplit], data: np.ndarray, val_size: float | int) -> None:
        """Test split with train and val sets."""
        TestSubsetsSplit._test_train_other_set(TestSubsetsSplit._get_split(splits_fn), VAL_SET, val_size, data)

    @staticmethod
    @pytest.mark.parametrize("val_size", [None, 0])
    @pytest.mark.parametrize("test_size", [0.1, 20])
    def test_train_test(splits_fn: Callable[[], DatasetSplit], data: np.ndarray, test_size: float | int) -> None:
        """Test split with train and test sets."""
        TestSubsetsSplit._test_train_other_set(TestSubsetsSplit._get_split(splits_fn), TEST_SET, test_size, data)

    @staticmethod
    @pytest.mark.parametrize("val_size", [0.2, 10])
    @pytest.mark.parametrize("test_size", [0.1, 20])
    def test_train_val_test(
        splits_fn: Callable[[], DatasetSplit], data: np.ndarray, val_size: float | int, test_size: float | int
    ) -> None:
        """Test split with train, val, and test sets."""
        split = TestSubsetsSplit._get_split(splits_fn)
        # Ensure that all train, val and test sets are present
        assert {TRAIN_SET, VAL_SET, TEST_SET} == split.keys()
        # Ensure that the train, val and test sets are disjoint
        assert set(split[TRAIN_SET]).isdisjoint(split[VAL_SET])
        assert set(split[TRAIN_SET]).isdisjoint(split[TEST_SET])
        assert set(split[VAL_SET]).isdisjoint(split[TEST_SET])
        # Ensure that the cardinalities of the val and test sets are correct
        val_card = val_size if isinstance(val_size, int) else int(val_size * len(data))
        assert len(set(split[VAL_SET])) == val_card
        test_card = test_size if isinstance(test_size, int) else int(test_size * len(data))
        assert len(set(split[TEST_SET])) == test_card
        # Ensure that the train, val and test sets cover the full dataset
        assert len(set(split[TRAIN_SET]) | set(split[VAL_SET]) | set(split[TEST_SET])) == len(data)


@pytest.mark.parametrize(
    ("split_fn", "expected_serialization"),
    [
        (
            k_fold,
            "k_fold(holdout_test_size=None,n_splits=10,random_state=12345,shuffle=True,stratify={stratify},stratify_bins=10,test_fold=True)",
        ),
        (
            subsets_split,
            "subsets_split(random_state=12345,stratify={stratify},stratify_bins=10,test_size=0.2,val_size=0.1)",
        ),
    ],
)
@pytest.mark.parametrize("stratify", [True, False])
def test_serialization(
    split_fn: Callable[[Sequence[Any], Sequence[int] | None, ...], DatasetSplit],
    stratify: bool,
    expected_serialization: str,
) -> None:
    """Test the string serialization of the split functions with default parameters.

    Args:
        split_fn: The split function for which to test the serialization.
        stratify: Whether to split the data in a stratified fashion, using the dataset's class labels.
        expected_serialization: The expected string serialization, with a replacement field for `stratify`.
    """
    assert serialize_split_fn(split_fn, stratify) == expected_serialization.format(stratify=stratify)
