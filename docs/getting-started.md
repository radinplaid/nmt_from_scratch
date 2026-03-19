# Getting Started

To get started with `quickmt-train`, you'll first need to install the library.

## Installation

You can install `quickmt-train` directly from source:

```bash
pip install -e .
```

To install the documentation dependencies, run:

```bash
pip install -e ".[docs]"
```

## Basic Usage

The library provides several command-line tools for training and evaluating NMT models.

### Training

To begin with you'll need a config file. See the [configs](configs/) directory for examples. See the [config.py](api-reference.md#config) module for more information on the config file format and default values.

Once you have a config file you're happy with you can start training a model:

```bash
quickmt-train --config configs/faen-tiny.yaml
```

### Model Averaging

To average the model weights from multiple checkpoints, use the `quickmt-avg` command:

```bash
quickmt-avg --experiment_dir ./faen-tiny
```

### Model Export

To export the model to CTranslate2 format, use the `quickmt-export` command:

```bash
quickmt-export --experiment_dir ./faen-tiny
```

### Evaluation

To evaluate a trained model, use the `quickmt-eval` command:

```bash
quickmt-eval --model_path output/model.safetensors --config configs/faen-tiny.yaml
```
