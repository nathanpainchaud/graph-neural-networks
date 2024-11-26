<div align="center">

# Your Project Name

[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![lightning-hydra-template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/nathanpainchaud/lightning-hydra-template)
<br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/nathanpainchaud/graph-neural-networks
cd graph-neural-networks

# [OPTIONAL] create virtual environment
python -m venv ./venv
source ./venv/bin/activate

# install project
# you must specify as an extra the desired CPU/CUDA versions of PyTorch
# Supported values are: cpu, cu124, cu121, cu118
# [OPTIONAL] you can also specify other extras for more functionalities
# Supported values are: wandb (for W&B integration)
pip install -e .[cu124,wandb]
```

#### uv

> [uv](https://docs.astral.sh/uv/) is a Python package and project manager.
> It allows you to manage Python interpreters, dependencies, and project configuration in a single tool.
> If you don't have it installed already, you can install it (on Linux and macOS) by running:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

```bash
# clone project
git clone https://github.com/YourGithubName/graph-neural-networks
cd graph-neural-networks

# create uv environment
# you must specify as an extra the desired CPU/CUDA versions of PyTorch
# Supported values are: cpu, cu124, cu121, cu118
# [OPTIONAL] you can also specify other extras for more functionalities
# Supported values are: wandb (for W&B integration)
uv sync --extra cu124 --extra wandb
source ./venv/bin/activate
```

## How to run

Train model with default configuration

```bash
# train on CPU
gnn-train trainer=cpu

# train on GPU
gnn-train trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](src/graph_neural_networks/configs/experiment/)

```bash
gnn-train experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
gnn-train trainer.max_epochs=20 data.batch_size=64
```
