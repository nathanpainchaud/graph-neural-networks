<div align="center">

# Graph Neural Networks

[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/pytorch-lightning)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![lightning-hydra-template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/nathanpainchaud/lightning-hydra-template)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://renovatebot.com/)
<br>
[![code-quality](https://github.com/nathanpainchaud/graph-neural-networks/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/nathanpainchaud/graph-neural-networks/actions/workflows/code-quality-main.yaml)
[![tests](https://github.com/nathanpainchaud/graph-neural-networks/actions/workflows/tests.yaml/badge.svg)](https://github.com/nathanpainchaud/graph-neural-networks/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/nathanpainchaud/graph-neural-networks/branch/main/graph/badge.svg?token=O4Y91WFUI5)](https://codecov.io/gh/nathanpainchaud/graph-neural-networks)
<br>
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/nathanpainchaud/graph-neural-networks?tab=Apache-2.0-1-ov-file)

# Publications

</div>

## Description

A template project for training graph neural networks using PyTorch Geometric, Lightning and Hydra. It tries to minimize
the complexity of the boilerplate code and configuration management, so that they can be easily understood and modified
to suit your needs, while still providing a feature-complete and flexible framework for working with GNNs.

> [!IMPORTANT]
> Using this template requires a basic understanding of PyTorch Lightning and Hydra. If you do not know at least what
> these libraries do and how they work at a high level, you should familiarize yourself with them.
> We refer you to the [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) and the
> [Hydra documentation](https://hydra.cc/docs/intro/).

## Installation

#### uv (recommended)

> [!NOTE]
> [uv](https://docs.astral.sh/uv/) is a Python package and project manager.
> It allows you to manage Python interpreters, dependencies, and project configuration in a single tool.
> If you don't have it installed already, you can install it (on Linux and macOS) by running:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

1. Download the repository.
   ```bash
   git clone https://github.com/nathanpainchaud/graph-neural-networks
   cd graph-neural-networks
   ```
2. Create a virtual environment and install the project and its dependencies. You must specify as an extra the desired
   compute platform for PyTorch (i.e. CPU/CUDA). Supported values are: `cpu`, `cu124`, `cu121`, `cu118`.
   ```bash
   # e.g. to install the project with the PyTorch version built for CPU
   uv sync --extra cpu

   # e.g. to install the project with the PyTorch version built for CUDA 12.4
   uv sync --extra cu124
   ```
   [OPTIONAL] You can also specify other extras for additional functionalities:
   ```bash
   # e.g. to install the `wandb` and `tensorboard` extras for loggers' integration
   uv sync --extra cpu --extra wandb --extra tensorboard

   # e.g. to install the `ogb` extra for Open Graph Benchmark datasets
   uv sync --extra cpu --extra ogb

   # e.g. to install all extra functionalities at once
   uv sync --extra cpu --extra all
   ```
3. Activate the virtual environment created by `uv`.
   ```bash
   source .venv/bin/activate
   ```

#### Pip

1. Download the repository.
   ```bash
   git clone https://github.com/nathanpainchaud/graph-neural-networks
   cd graph-neural-networks
   ```
2. Create a virtual environment and activate it.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install PyTorch libraries (i.e. `torch` and `torchvision`) according to the [official instructions](https://pytorch.org/get-started/locally/).
   Follow the instructions for `pip` and the compute platform compatible with your system.
   ```bash
   # e.g. to install the PyTorch version built for CPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # e.g. to install the PyTorch version built for CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
4. Install the project in editable mode.
   ```bash
   pip install -e .
   ```
   [OPTIONAL] You can also specify other extras for additional functionalities:
   ```bash
   # e.g. to install the `wandb` and `tensorboard` extras for loggers' integration
   pip install -e .[wandb,tensorboard]

   # e.g. to install the `ogb` extra for Open Graph Benchmark datasets
   pip install -e .[ogb]

   # e.g. to install all extra functionalities at once
   pip install -e .[all]
   ```

### List of available extras

- \[`cpu`|`cu124`|`cu121`|`cu118`\]: Required mutually exclusive extras to install the project with a PyTorch version
  built for CPU or a specific CUDA version (only available when using `uv`, not `pip`).
- `wandb`: For experiment tracking with Weights & Biases.
- `tensorboard`: For experiment tracking with TensorBoard.
- `ogb`: For Open Graph Benchmark datasets.
- `all`: Install all (non-mutually exclusive) extras at once.

### Setup Weight & Biases

#### Create an account

Follow the instructions on the [Weights & Biases website](https://docs.wandb.ai/quickstart#1-create-an-account-and-install-wb)
to create an account.

#### Install W&B

Make sure that you install the `wandb` extra when installing the project, as shown in the [installation instructions](#installation).

#### Configure your credentials

The recommended way to configure your W&B credentials is to expose them as environment variables
(see [W&B's documentation on this](https://docs.wandb.ai/guides/track/environment-variables/)). You can do this by
copying the [`configs/local/example.yaml`](src/graph_neural_networks/configs/local/example.yaml) to a new `default.yaml`
(which will be ignored by Git) and filling in your W&B credentials.

You don't have to do anything more than that, as the project is configured to automatically load keys under `hydra.job.env_set`
as environment variables when executing the scripts.

#### Use the wandb logger

Follow the instructions provided in the [How to run](#track-experiments) section to enable experiment tracking via W&B.

## How to run

### The basics

Train model with the default configuration.

> [!WARNING]
> The default configuration is not complete and running it as-is will fail, asking you to specify the missing `data`
> and `model` groups.

```bash
# train on CPU
gnn-train trainer=cpu data=<YOUR_DATA_CONFIG> data.dataset=<YOUR_DATASET_CONFIG> model=<YOUR_MODEL_CONFIG>

# train on GPU
gnn-train trainer=gpu data=<YOUR_DATA_CONFIG> data.dataset=<YOUR_DATASET_CONFIG> model=<YOUR_MODEL_CONFIG>
```

Override any individual parameter in the config files from the command line like this:

```bash
gnn-train trainer.max_epochs=20 data.batch_size=64 ...
```

### Use preset configs

Train model with chosen experiment configuration from [configs/experiment/](src/graph_neural_networks/configs/experiment/).

> [!TIP]
> This allows you to provide (complete) presets on top of the default configuration, typically for experiments you want
> to run regularly.

```bash
gnn-train experiment=<YOUR_EXPERIMENT_CONFIG>
```

### Track experiments

Although some basic logging configurations are provided for CSV file and TensorBoard, the recommended tool to track
experiments is [Weights & Biases](https://wandb.ai/site), by using W&B's
[integration in PyTorch Lightning](https://docs.wandb.ai/guides/integrations/lightning/).

> [!WARNING]
> You must have followed the [W&B setup instructions](#setup-weight--biases) to use this feature.

```bash
# track experiment locally w/ TensorBoard
gnn-train experiment=<YOUR_EXPERIMENT_CONFIG> logger=tensorboard
# and in another terminal
tensorboard --logdir ./logs/

# track experiment online w/ W&B
gnn-train experiment=<YOUR_EXPERIMENT_CONFIG> logger=wandb

# track experiment offline w/ W&B
gnn-train experiment=<YOUR_EXPERIMENT_CONFIG> logger=wandb logger.wandb.offline=True

# track experiments using different loggers at once (i.e. CSV, TensorBoard, W&B)
gnn-train experiment=<YOUR_EXPERIMENT_CONFIG> logger=many_loggers
```

### Run multiple experiments

Launch multiple experiments at once using the `multirun` (`-m`) option.

```bash
# run multiple experiments sequentially, here w/ 5 different seeds
gnn-train -m experiment=<YOUR_EXPERIMENT_CONFIG> seed=0,1,2,3,4
```

Launch multiple experiments at once **in parallel** using the [Joblib launcher for Hydra](https://hydra.cc/docs/plugins/joblib_launcher/).

> [!NOTE]
> The `hydra-joblib-launcher` plugin required to use this feature is installed by default with the project, so no need
> to install it by yourself.

```bash
# run multiple experiments in parallel, here w/ 5 different seeds
gnn-train -m hydra/launcher=joblib experiment=<YOUR_EXPERIMENT_CONFIG> seed=0,1,2,3,4
```

### Run automatic hyperparameter search with Optuna

Launch an automatic hyperparameter search using the [Optuna sweeper for Hydra](https://hydra.cc/docs/plugins/optuna_sweeper/).

> [!WARNING]
> You have to make sure that the `hparams_search` config you use is compatible with the model, since `hparams_search`
> defines how to sweep over model-dependent config options.

```bash
# Example of a predefined Optuna config for graph-level models with a compatible experiment
gnn-train experiment=graph_classification hparams_search=graph_level_optuna
```

> [!TIP]
> Optuna can be used in a cross-validation setting, by evaluating each sampling of hyperparameters on the different
> dataset folds and reporting the average performance. However, this approach is not compatible with the default Optuna
> sweeper plugin, where each trial corresponds to one Hydra run, i.e. one model trained/evaluated on a specific
> partition of the dataset.
>
> To support this feature, we rely on our custom `serial_sweeper`, designed to run multiple jobs in sequence within the
> same Hydra run and then aggregate the results of these jobs. By sweeping over the different folds with this sweeper,
> we support cross-validation with Optuna.
>
> This is all handled already in the predefined Optuna config `graph_level_optuna` for graph-level models. However, if
> you want to support this in your own Optuna config, all you have to do is to use the predefined `cross_validation`
> config for `serial_sweeper`, and make sure that `data.split=kfold` is used to split the data into multiple folds.
>
> ```bash
> gnn-train [...] hparams_search=<YOUR_OPTUNA_CONFIG> data.split=kfold serial_sweeper=cross_validation
> ```

### Run tests

Run the tests using [Pytest](https://docs.pytest.org/en/stable/).

```bash
# run all tests
pytest

# run a test package
pytest tests/integration

# run tests from a specific file
pytest tests/integration/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```
