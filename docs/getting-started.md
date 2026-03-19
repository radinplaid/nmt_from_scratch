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

To start a training session, use the `quickmt-train` command:

```bash
quickmt-train --config configs/faen-tiny.yaml
```

### Evaluation

To evaluate a trained model, use the `quickmt-eval` command:

```bash
quickmt-eval --model_path output/model.safetensors --config configs/faen-tiny.yaml
```
