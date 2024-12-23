import importlib.metadata
import sys
from typing import Any

import pytest
import torch
from packaging.version import Version
from pytest import MarkDecorator

from .package_available import _IS_WINDOWS, _OGB_AVAILABLE, _SH_AVAILABLE, _WANDB_AVAILABLE, _XLA_AVAILABLE

if _SH_AVAILABLE:
    import sh


class RunIf:
    """RunIf wrapper for conditional skipping of tests.

    Fully compatible with `@pytest.mark`.

    Example:
    ```python
        @RunIf(min_torch="1.8")
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        min_torch: str | None = None,
        max_torch: str | None = None,
        min_python: str | None = None,
        skip_windows: bool = False,
        sh: bool = False,
        xla: bool = False,
        wandb: bool = False,
        ogb: bool = False,
        **kwargs: dict[Any, Any],
    ) -> MarkDecorator:
        """Creates a new `@RunIf` `MarkDecorator` decorator.

        Args:
            min_gpus: Min number of GPUs required to run test.
            min_torch: Minimum pytorch version to run test.
            max_torch: Maximum pytorch version to run test.
            min_python: Minimum python version required to run test.
            skip_windows: Skip test for Windows platform.
            xla: If XLA is available.
            sh: If `sh` module is required to run the test.
            wandb: If `wandb` module is required to run the test.
            ogb: If `ogb` module is required to run the test.
            **kwargs: Native `pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = importlib.metadata.version("torch")
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = importlib.metadata.version("torch")
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if skip_windows:
            conditions.append(_IS_WINDOWS)
            reasons.append("does not run on Windows")

        if xla:
            conditions.append(not _XLA_AVAILABLE)
            reasons.append("XLA")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh is not supported on Windows")

        if wandb:
            conditions.append(not _WANDB_AVAILABLE)
            reasons.append("wandb")

        if ogb:
            conditions.append(not _OGB_AVAILABLE)
            reasons.append("ogb")

        reasons = [rs for cond, rs in zip(conditions, reasons, strict=False) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )


def run_sh_command(command: list[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    Args:
        command: A list of shell commands as strings.
    """
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(reason=msg)
