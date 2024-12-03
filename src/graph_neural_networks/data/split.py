from numpy.random import RandomState
from sklearn import model_selection
from torch_geometric.data import Dataset

DatasetSplit = list[dict[str, list[int]]]


def multigraph_k_fold(
    dataset: Dataset,
    n_splits: int = 10,
    val_size: float | int | None = 0.1,
    shuffle: bool = True,
    stratify: bool = False,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a multi-graph dataset into k folds for cross-validation, with each graph treated as a sample.

    :param dataset: The multi-graph dataset to split.
    :param n_splits: The number of folds to create.
    :param val_size: The size of the validation set to additionally create from the training set.
        If `None`, no validation set is created.
    :param shuffle: Whether to shuffle the data before splitting.
    :param stratify: Whether to stratify the data based on the graph-level target labels.
    :param random_state: The random state to use for reproducibility.

    :return: A list of splits, where each split contains the indices of its train, test and (optional) val sets.
    """
    k_fold_cls = model_selection.StratifiedKFold if stratify else model_selection.KFold
    k_fold = k_fold_cls(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = []
    for train_idx, test_idx in k_fold.split(dataset.indices(), y=dataset.y if stratify else None):
        # Convert the int64 arrays returned by sklearn's KFold to lists of base integers
        train_idx, test_idx = train_idx.astype(int).tolist(), test_idx.astype(int).tolist()  # noqa: PLW2901

        if val_size:
            # If `val_size` is provided, split the training set into training and validation sets
            train_subset = dataset[train_idx]
            split_kwargs = {"test_size": val_size, "shuffle": shuffle, "random_state": random_state}
            if stratify:
                split_kwargs["stratify"] = train_subset.y

            train_idx, val_idx = model_selection.train_test_split(train_subset.indices(), **split_kwargs)  # noqa: PLW2901

        current_fold = {"train": train_idx, "test": test_idx}
        if val_size:
            current_fold["val"] = val_idx
        folds.append(current_fold)

    return folds


def multigraph_split(
    dataset: Dataset,
    test_size: float | int = 0.2,
    val_size: float | int | None = 0.1,
    stratify: bool = False,
    shuffle: bool = True,
    random_state: int | RandomState | None = 12345,
) -> DatasetSplit:
    """Splits a multi-graph dataset into train, test and (optional) val sets, with each graph treated as a sample.

    :param dataset: The multi-graph dataset to split.
    :param test_size: The size of the test set.
    :param val_size: The size of the validation set to additionally create from the training set.
        If `None`, no validation set is created.
    :param stratify: Whether to stratify the data based on the graph-level target labels.
    :param shuffle: Whether to shuffle the data before splitting.
    :param random_state: The random state to use for reproducibility.

    :return: A list containing a single split between the train, test and (optional) val sets.
    """
    test_split_kwargs = {"test_size": test_size, "shuffle": shuffle, "random_state": random_state}
    if stratify:
        test_split_kwargs["stratify"] = dataset.y

    train_idx, test_idx = model_selection.train_test_split(dataset.indices(), **test_split_kwargs)

    if val_size:
        # Create a training subset that excludes the test set
        train_subset = dataset[train_idx]
        val_split_kwargs = {"test_size": val_size, "shuffle": shuffle, "random_state": random_state}
        if stratify:
            val_split_kwargs["stratify"] = train_subset.y

        train_idx, val_idx = model_selection.train_test_split(train_subset.indices(), **val_split_kwargs)

    split = {"train": train_idx, "test": test_idx}
    if val_size:
        split["val"] = val_idx

    return [split]
