import functools
from collections.abc import Callable
from pathlib import Path

import pytest
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset


@pytest.fixture
def mutag_dataset_fn(tmp_path: Path) -> Callable[[], Dataset]:
    """A Pytest fixture for a function that returns a `TUDataset` instance for the MUTAG dataset.

    Args:
        tmp_path: The temporary data path.

    Returns:
        A function that returns a dataset.
    """
    return functools.partial(TUDataset, root=str(tmp_path), name="MUTAG")


@pytest.fixture
def ogbg_molhiv_dataset_fn(tmp_path: Path) -> Callable[[], Dataset]:
    """A Pytest fixture for a function that returns a PyG `Dataset` instance for the OGB-MOLHIV dataset.

    Args:
        tmp_path: The temporary data path.

    Returns:
        A function that returns a dataset.
    """
    from ogb.graphproppred import PygGraphPropPredDataset

    return functools.partial(PygGraphPropPredDataset, root=str(tmp_path), name="ogbg-molhiv")
