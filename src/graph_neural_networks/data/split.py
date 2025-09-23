import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from lightning_utilities import apply_to_collection
from numpy.random import RandomState
from sklearn import model_selection

from graph_neural_networks.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


DatasetSplit = list[dict[str, list[int]]]
TRAIN_SET = "train"
VAL_SET = "val"
TEST_SET = "test"


def k_fold(
    data: Sequence[Any],
    stratify: Sequence[int | float] | None = None,
    stratify_bins: int | Sequence[float] | None = None,
    n_splits: int = 10,
    test_fold: bool = True,
    holdout_test_size: float | int | None = None,
    shuffle: bool = True,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a sequence of sample indices into k folds for cross-validation.

    Args:
        data: Data of length `n_samples` to split.
        stratify: If provided, the data is split in a stratified fashion, using this as the class labels.
        stratify_bins: The bins to divide the stratify labels into, assuming they represent a continuous target.
            If `bins` is an int, it defines the number of equal-width bins in the given range.
            If bins is a sequence, it defines a monotonically increasing array of bin edges allowing for non-uniform bin
            widths (see `bins` argument of `np.digitize`)
        n_splits: The number of folds/splits to create.
        holdout_test_size: The size of the holdout test set to split from the training set before creating the K folds.
            This effectively means that all splits are assigned the same test set. If None or 0, the full dataset is
            used when creating the folds. Mutually exclusive with `test_fold`.
        test_fold: Whether to reserve a fold for testing in each of the splits, using a separate fold for validation
            and K-2 folds for training. If None or False, use one fold for validation/evaluation and K-1 folds for
            training. Mutually exclusive with `holdout_test_size`.
        shuffle: Whether to shuffle the data before splitting.
        random_state: The random state to use for reproducibility.

    Returns:
        A list of splits, where each split contains the indices of its train, val and (optional) test sets.
    """
    if holdout_test_size and test_fold:
        raise ValueError(
            "You specified both `holdout_test_size` and `test_fold`, which are mutually exclusive strategies for "
            "defining a test set. Pick one of the two."
        )

    # Create an array of indices relative to the input data +
    # ensure that labels are a numpy array to easily index them using arrays
    indices = np.arange(len(data))
    if stratify is not None:
        stratify = np.array(stratify)
        # Support continuous labels, by dividing them into discrete bins that sklearn can use for stratification
        if stratify_bins:
            stratify = _digitize_labels(stratify, stratify_bins)

    if holdout_test_size:
        test_split_cls = model_selection.ShuffleSplit if stratify is None else model_selection.StratifiedShuffleSplit
        test_split = test_split_cls(n_splits=1, test_size=holdout_test_size, random_state=random_state)
        train_idx, test_idx = next(test_split.split(indices, stratify))

        # Excludes the hold-out test set from the data
        indices = indices[train_idx]
        if stratify is not None:
            stratify = stratify[train_idx]

    k_fold_cls = model_selection.KFold if stratify is None else model_selection.StratifiedKFold
    k_fold = k_fold_cls(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = []
    for train_idx, val_idx in k_fold.split(indices, y=stratify):
        split = {TRAIN_SET: indices[train_idx], VAL_SET: indices[val_idx]}
        if holdout_test_size:
            split[TEST_SET] = test_idx
        splits.append(split)

    if test_fold:
        for split_idx, split in enumerate(splits):
            # Use the validation fold of the previous split as the current split's test fold
            split[TEST_SET] = splits[split_idx - 1][VAL_SET]
            # Remove the newly assigned test samples from the training set
            split[TRAIN_SET] = np.setdiff1d(split[TRAIN_SET], split[TEST_SET], assume_unique=True)

    # Convert arrays of int64 (e.g. returned by `KFold.split`) to a sorted native int list
    # to avoid serialization issues if the caller tries to save the splits to disk
    return apply_to_collection(splits, np.ndarray, lambda x: np.sort(x).tolist())


def subsets_split(
    data: Sequence[Any],
    stratify: Sequence[int | float] | None = None,
    stratify_bins: int | Sequence[float] | None = None,
    val_size: float | int | None = 0.1,
    test_size: float | int | None = 0.2,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a sequence of sample indices into train, and optional val and test sets.

    Args:
        data: Data of length `n_samples` to split.
        stratify: If provided, the data is split in a stratified fashion, using this as the class labels.
        stratify_bins: The bins to divide the stratify labels into, assuming they represent a continuous target.
            If `bins` is an int, it defines the number of equal-width bins in the given range.
            If bins is a sequence, it defines a monotonically increasing array of bin edges allowing for non-uniform bin
            widths (see `bins` argument of `np.digitize`)
        val_size: The size of the validation set. If None or 0, no validation set is created.
        test_size: The size of the test set. If None or 0, no test set is created.
        random_state: The random state to use for reproducibility.

    Returns:
        A list containing a single split between the train, and optional val and test sets.
    """
    if not (val_size or test_size):
        log.warning(
            "No validation or test set configured. Returning the full dataset as train set. This is likely a mistake "
            "and not intended behavior."
        )

    # Create an array of indices relative to the input data +
    # ensure that labels are a numpy array to easily index them using arrays
    indices = np.arange(len(data))
    if stratify is not None:
        stratify = np.array(stratify)
        # Support continuous labels, by dividing them into discrete bins that sklearn can use for stratification
        if stratify_bins:
            stratify = _digitize_labels(stratify, stratify_bins)

    split = {TRAIN_SET: indices}

    splits_cls = model_selection.ShuffleSplit if stratify is None else model_selection.StratifiedShuffleSplit

    if val_size:
        val_split = splits_cls(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(val_split.split(indices, stratify))
        split[TRAIN_SET], split[VAL_SET] = indices[train_idx], indices[val_idx]

    if test_size:
        # Make sure to exclude the val set from the data, if one was created
        indices = indices[split[TRAIN_SET]]
        if stratify is not None:
            stratify = stratify[split[TRAIN_SET]]

        # If `test_size` represents a proportion of the dataset, compute the proportion relative to the training set
        # (to adjust the proportion in case a val set was split)
        if isinstance(test_size, float):
            test_size *= len(data) / len(indices)

        test_split = splits_cls(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(test_split.split(indices, stratify))
        split[TRAIN_SET], split[TEST_SET] = indices[train_idx], indices[test_idx]

    # Convert arrays of int64 (e.g. returned by `ShuffleSplit.split`) to a sorted native int list
    # to avoid serialization issues if the caller tries to save the splits to disk
    split = apply_to_collection(split, np.ndarray, lambda x: np.sort(x).tolist())

    return [split]


def serialize_split_fn(
    split_fn: Callable[[Sequence[Any], Sequence[int] | None, ...], DatasetSplit], stratify: bool
) -> str:
    """Serialize a split function to a string to use as a unique identifier for the splits.

    Args:
        split_fn: The split function to serialize.
        stratify: Whether the function will be provided supervised labels to create splits in a stratified fashion.

    Returns:
        A unique string representation of the split function.
    """
    # Inspect the signature of the split function to get the default parameters
    split_params = inspect.signature(split_fn).parameters.copy()
    split_params.popitem(last=False)  # Del the first param passed to `split_fn` (the positional dataset arg)
    split_params.pop("stratify", None)  # Del the `stratify` param, as it's overridden by the flag
    split_params = {k: v.default for k, v in split_params.items()}  # Unpack the `Signature` object
    split_params["stratify"] = stratify  # Add the stratify flag to the params

    # Sort the params to ensure a consistent repr, even if the order of the params changes
    params_repr = ",".join(f"{k}={v}" for k, v in sorted(split_params.items()))
    split_fn_name = split_fn.func.__name__ if isinstance(split_fn, functools.partial) else split_fn.__name__
    return f"{split_fn_name}({params_repr})"


def _digitize_labels(labels: Sequence[int | float], bins: int | Sequence[float]) -> np.ndarray:
    """Digitizes continuous labels, e.g. regression targets, into bins to help create stratified splits.

    Args:
        labels: The labels to digitize.
        bins: The number of bins to create, or directly the bins to use.

    Returns:
        An array of bin indices for each label.
    """
    if isinstance(bins, int):
        # `np.digitize` assigns 0/len(bins) to values beyond the left/right bins' bounds, respectively.
        # Thus, to naturally get indices in the range [0, bins-1], we exclude the min/max from the bins by:
        # 1) setting endpoint=False in linspace to exclude the max value
        # 2) removing the first bin edge to exclude the min value
        bins = np.linspace(np.min(labels), np.max(labels), num=bins, endpoint=False)[1:]
    return np.digitize(labels, bins)
