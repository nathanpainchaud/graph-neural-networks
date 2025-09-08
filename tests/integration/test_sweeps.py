import os
from pathlib import Path

import pytest

from ..helpers.run import RunIf, run_sh_command  # noqa: TID252


@pytest.fixture(scope="module")
def train_script() -> Path:
    """A pytest fixture for the training script.

    Returns:
        The path to the training script.
    """
    # This has to be a fixture rather than a module-level variable because it relies on the PROJECT_ROOT env var
    # having been set by another session-scoped fixture
    return Path(os.environ["PROJECT_ROOT"], "src/graph_neural_networks/train.py")


@pytest.fixture
def testing_overrides(tmp_path: Path) -> list[str]:
    """A pytest fixture for the overrides to use for tests (e.g. logging, accelerator, compilation, etc.).

    Returns:
        A list of configuration overrides.
    """
    return [
        "hydra.run.dir=" + str(tmp_path),
        "hydra.sweep.dir=" + str(tmp_path),
        "logger=[]",
        # Uncomment the following line if compilation is enabled by default to disable it for tests
        # "compile=false",  # Disable compilation to speed up tests
    ]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(train_script: Path, testing_overrides: list[str]) -> None:
    """Test running all available experiment configs with `fast_dev_run=True`.

    Args:
        train_script: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(train_script),
        "-m",
        "experiment=glob(*)",
        "++trainer.fast_dev_run=true",
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(train_script: Path, application_overrides: list[str], testing_overrides: list[str]) -> None:
    """Test default hydra sweep.

    Args:
        train_script: The path of the script to invoke.
        application_overrides: The application-specific overrides to use.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(train_script),
        "-m",
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
        *application_overrides,
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(
    train_script: Path, mutag_classification_overrides: list[str], testing_overrides: list[str]
) -> None:
    """Test Optuna hyperparam sweeping.

    Args:
        train_script: The path of the script to invoke.
        mutag_classification_overrides: The overrides to use for the MUTAG classification task.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    # TODO: Make this test more flexible by using `application_overrides` (instead of `mutag_classification_overrides`)
    #       and automatically picking the `hparams_search` config based on the application overrides.
    #       Not done because it is not trivial to find a reliable way to pick an appropriate `hparams_search` config.
    command = [
        str(train_script),
        "-m",
        "hparams_search=graph_classification_optuna",
        "~serial_sweeper",  # Disable the serial sweeper here to test it separately
        "hydra.sweeper.n_jobs=1",
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
        *mutag_classification_overrides,
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_serial_sweep(train_script: Path, application_overrides: list[str], testing_overrides: list[str]) -> None:
    """Test single-process serial sweeping.

    Args:
        train_script: The path of the script to invoke.
        application_overrides: The application-specific overrides to use.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(train_script),
        "serial_sweeper=cross_validation",
        "++trainer.fast_dev_run=true",
        *application_overrides,
        *testing_overrides,
    ]
    run_sh_command(command)
