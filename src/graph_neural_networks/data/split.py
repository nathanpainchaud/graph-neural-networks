from numpy.random import RandomState
from sklearn import model_selection
from torch_geometric.data import Dataset

from graph_neural_networks.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


DatasetSplit = list[dict[str, list[int]]]
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"


def intergraph_k_fold(
    dataset: Dataset,
    test_fold: bool = True,
    holdout_test_size: float | int | None = None,
    n_splits: int = 10,
    shuffle: bool = True,
    stratify: bool = False,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a multi-graph dataset into k folds for cross-validation, with each graph treated as a sample.

    Args:
        dataset: The multi-graph dataset to split.
        holdout_test_size: The size of the holdout test set to split from the training set before creating the K folds.
            This effectively means that all splits are assigned the same test set. If None or 0, the full dataset is
            used when creating the folds. Mutually exclusive with `test_fold`.
        test_fold: Whether to reserve a fold for testing in each of the splits, using a separate fold for validation
            and K-2 folds for training. If None or False, use one fold for validation/evaluation and K-1 folds for
            training. Mutually exclusive with `holdout_test_size`.
        n_splits: The number of folds/splits to create.
        shuffle: Whether to shuffle the data before splitting.
        stratify: Whether to stratify the data based on the graph-level target labels.
        random_state: The random state to use for reproducibility.

    Returns:
        A list of splits, where each split contains the indices of its train, val and (optional) test sets.
    """
    if holdout_test_size and test_fold:
        raise ValueError(
            "You specified both `holdout_test_size` and `test_fold`, which are mutually exclusive strategies for "
            "defining a test set. Pick one of the two."
        )

    k_fold_cls = model_selection.StratifiedKFold if stratify else model_selection.KFold
    k_fold = k_fold_cls(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    if holdout_test_size:
        test_split_kwargs = {"test_size": holdout_test_size, "shuffle": shuffle, "random_state": random_state}
        if stratify:
            test_split_kwargs["stratify"] = dataset.y

        holdout_train_idx, test_idx = model_selection.train_test_split(dataset.indices(), **test_split_kwargs)

    # Create a subset that excludes the test set (if it exists)
    k_fold_data = dataset[holdout_train_idx] if holdout_test_size else dataset
    splits = []
    for train_idx, val_idx in k_fold.split(k_fold_data.indices(), y=k_fold_data.y if stratify else None):
        # Convert the int64 arrays returned by sklearn's KFold to lists of base integers
        train_idx, val_idx = train_idx.astype(int).tolist(), val_idx.astype(int).tolist()  # noqa: PLW2901

        split = {TRAIN_SPLIT: train_idx, VAL_SPLIT: val_idx}
        if holdout_test_size:
            split[TEST_SPLIT] = test_idx

        splits.append(split)

    # When test folds are requested:
    # 1) use the validation fold of the previous split as the current split's test fold
    # 2) remove the newly assigned test samples from the training set
    if test_fold:
        for split_idx, split in enumerate(splits):
            split[TEST_SPLIT] = splits[split_idx - 1][VAL_SPLIT]
            split[TRAIN_SPLIT] = list(set(split[TRAIN_SPLIT]) - set(split[TEST_SPLIT]))

    return splits


def intergraph_split(
    dataset: Dataset,
    val_size: float | int | None = 0.1,
    test_size: float | int | None = 0.2,
    stratify: bool = False,
    shuffle: bool = True,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a multi-graph dataset into train, and optional val and test sets, with each graph treated as a sample.

    Args:
        dataset: The multi-graph dataset to split.
        val_size: The size of the validation set. If None or 0, no validation set is created. If both `test_size` and
            `val_size` are provided, creates the test set first, and then the validation set from the remaining training
            set.
        test_size: The size of the test set. If None or 0, no test set is created. If both `test_size` and `val_size`
            are provided, creates the test set first, and then the validation set from the remaining training set.
        stratify: Whether to stratify the data based on the graph-level target labels.
        shuffle: Whether to shuffle the data before splitting.
        random_state: The random state to use for reproducibility.

    Returns:
        A list containing a single split between the train, and optional val and test sets.
    """
    if not (val_size or test_size):
        log.warning(
            "No validation or test set configured. Returning the full dataset as train set. This is likely a mistake "
            "and not intended behavior."
        )

    split = {TRAIN_SPLIT: list(dataset.indices())}

    if test_size:
        # Include the full dataset in the split
        test_split_kwargs = {"test_size": test_size, "shuffle": shuffle, "random_state": random_state}
        if stratify:
            test_split_kwargs["stratify"] = dataset.y

        split[TRAIN_SPLIT], split[TEST_SPLIT] = model_selection.train_test_split(dataset.indices(), **test_split_kwargs)

    # Create a subset that excludes the test set (if it exists)
    if val_size:
        data_to_split = dataset[split[TRAIN_SPLIT]]
        val_split_kwargs = {"test_size": val_size, "shuffle": shuffle, "random_state": random_state}
        if stratify:
            val_split_kwargs["stratify"] = data_to_split.y

        split[TRAIN_SPLIT], split[VAL_SPLIT] = model_selection.train_test_split(
            data_to_split.indices(), **val_split_kwargs
        )

    return [split]
