from numpy.random import RandomState
from sklearn import model_selection
from torch_geometric.data import Dataset

DatasetSplit = list[dict[str, list[int]]]
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"


def intergraph_k_fold(
    dataset: Dataset,
    test_size: float | int | None = None,
    n_splits: int = 10,
    shuffle: bool = True,
    stratify: bool = False,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a multi-graph dataset into k folds for cross-validation, with each graph treated as a sample.

    Args:
        dataset: The multi-graph dataset to split.
        test_size: The size of the holdout test set to split from the training set before creating the folds over the
            remaining training set. This effectively means that all folds "share" the same test set. If None or 0,
            no holdout test set is created.
        n_splits: The number of folds to create.
        shuffle: Whether to shuffle the data before splitting.
        stratify: Whether to stratify the data based on the graph-level target labels.
        random_state: The random state to use for reproducibility.

    Returns:
        A list of splits, where each split contains the indices of its train, val and (optional) test sets.
    """
    k_fold_cls = model_selection.StratifiedKFold if stratify else model_selection.KFold
    k_fold = k_fold_cls(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    if test_size:
        test_split_kwargs = {"test_size": test_size, "shuffle": shuffle, "random_state": random_state}
        if stratify:
            test_split_kwargs["stratify"] = dataset.y

        k_fold_idx, test_idx = model_selection.train_test_split(dataset.indices(), **test_split_kwargs)

    # Create a subset that excludes the test set (if it exists)
    k_fold_data = dataset[k_fold_idx] if test_size else dataset
    folds = []
    for fold_train_idx, fold_val_idx in k_fold.split(k_fold_data.indices(), y=k_fold_data.y if stratify else None):
        # Convert the int64 arrays returned by sklearn's KFold to lists of base integers
        fold_train_idx, fold_val_idx = fold_train_idx.astype(int).tolist(), fold_val_idx.astype(int).tolist()  # noqa: PLW2901

        current_fold = {TRAIN_SPLIT: fold_train_idx, VAL_SPLIT: fold_val_idx}
        if test_size:
            current_fold[TEST_SPLIT] = test_idx

        folds.append(current_fold)

    return folds


def intergraph_split(
    dataset: Dataset,
    val_size: float | int = 0.1,
    test_size: float | int | None = 0.2,
    stratify: bool = False,
    shuffle: bool = True,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a multi-graph dataset into train, and optional val and test sets, with each graph treated as a sample.

    Args:
        dataset: The multi-graph dataset to split.
        val_size: The size of the validation set. If both `test_size` and `val_size` are provided, the test set is
            created first, and the validation set is created from the remaining training set.
        test_size: The size of the test set. If None or 0, no test set is created.
        stratify: Whether to stratify the data based on the graph-level target labels.
        shuffle: Whether to shuffle the data before splitting.
        random_state: The random state to use for reproducibility.

    Returns:
        A list containing a single split between the train, and optional val and test sets.
    """
    split = {TRAIN_SPLIT: dataset.indices()}

    if test_size:
        # Include the full dataset in the split
        test_split_kwargs = {"test_size": test_size, "shuffle": shuffle, "random_state": random_state}
        if stratify:
            test_split_kwargs["stratify"] = dataset.y

        split[TRAIN_SPLIT], split[TEST_SPLIT] = model_selection.train_test_split(dataset.indices(), **test_split_kwargs)

    # Create a subset that excludes the test set (if it exists)
    data_to_split = dataset[split[TRAIN_SPLIT]]
    val_split_kwargs = {"test_size": val_size, "shuffle": shuffle, "random_state": random_state}
    if stratify:
        val_split_kwargs["stratify"] = data_to_split.y

    split[TRAIN_SPLIT], split[VAL_SPLIT] = model_selection.train_test_split(data_to_split.indices(), **val_split_kwargs)

    return [split]
