<div align="center">

# Graph Neural Networks

[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/pytorch-lightning)
[![hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![lightning-hydra-template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/nathanpainchaud/lightning-hydra-template)
<br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

A template project for training graph neural networks with PyTorch Lightning and Hydra. It tries to minimize the
complexity of the boilerplate code and configuration management, so that they can be easily understood and modified to
suit your needs, while still providing a feature-complete and flexible framework for working with GNNs.

> [!IMPORTANT]
> Using this template requires a basic understanding of PyTorch Lightning and Hydra. If you do not know at least what
> these libraries do and how they work at a high level, you should familiarize yourself with them.
> We refer you to the [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) and the
> [Hydra documentation](https://hydra.cc/docs/intro/).

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/nathanpainchaud/graph-neural-networks
cd graph-neural-networks

# [OPTIONAL] create virtual environment
python -m venv .venv
source .venv/bin/activate

# install project
# you must specify as an extra the desired CPU/CUDA versions of PyTorch
# Supported values are: cpu, cu124, cu121, cu118
# [OPTIONAL] you can also specify other extras for more functionalities
# Supported values are: wandb (for W&B integration)
pip install -e .[cu124,wandb]
```

#### uv

> [!NOTE]
> [uv](https://docs.astral.sh/uv/) is a Python package and project manager.
> It allows you to manage Python interpreters, dependencies, and project configuration in a single tool.
> If you don't have it installed already, you can install it (on Linux and macOS) by running:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

```bash
# clone project
git clone https://github.com/nathanpainchaud/graph-neural-networks
cd graph-neural-networks

# create uv environment
# you must specify as an extra the desired CPU/CUDA versions of PyTorch
# Supported values are: cpu, cu124, cu121, cu118
# [OPTIONAL] you can also specify other extras for more functionalities
# Supported values are: wandb (for W&B integration)
uv sync --extra cu124 --extra wandb
source .venv/bin/activate
```

### Setup Weight & Biases

#### Create an account

Follow the instructions on the [Weights & Biases website](https://docs.wandb.ai/quickstart#1-create-an-account-and-install-wb)
to create an account.

#### Install W&B

Make sure that you install the `wandb` extra when installing the project, as shown in the [installation instructions](#installation).

#### Configure your credentials

The recommended way to configure your W&B credentials is to expose them as environment variables
(see [W&B's documentation on this](https://docs.wandb.ai/guides/track/environment-variables/)). You can do this by
copying the [`.env.example`](.env.example) file to a new `.env` file (which will be ignored by Git) and filling in your
W&B credentials.

You don't have to do anything more than that, as the project is configured to automatically load environment variables
from the `.env` file when executing the scripts.

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
gnn-train trainer=cpu data=<YOUR_DATA_CONFIG> model=<YOUR_MODEL_CONFIG>

# train on GPU
gnn-train trainer=gpu data=<YOUR_DATA_CONFIG> model=<YOUR_MODEL_CONFIG>
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

```bash
gnn-train hparams_search=graph_classification_cv_optuna experiment=graph_classification
```

> [!TIP]
> Support for Optuna in a cross-validation setting is enabled by using the custom
> [`hydra_serial_sweeper`](src/graph_neural_networks/utils/utils.py#:~:text=hydra_serial_sweeper) decorator on the Hydra
> main function, along with the `serial_sweeper=cross_validation` option. This last option is already configured in the
> above [`graph_classification_cv_optuna`](src/graph_neural_networks/configs/hparams_search/graph_classification_cv_optuna.yaml)
> config.
