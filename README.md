
# ResNet from Scratch Lite

A minimal, educational implementation of ResNet architecture with residual blocks for CIFAR-10 classification.

## Features

- **ResidualBlock** (`resnet_block.py`): Core residual block with skip connections and batch normalization
- **MiniResNet** (`resnet_model.py`): Lightweight 3-layer ResNet model for CIFAR-10 (10 classes)
- **Training Pipeline** (`train_resnet.py`): Full training loop with CIFAR-10 dataset (10k subset) and metrics logging
- **Validation** (`validation.py`): Model evaluation utility

## Quick Start

```bash
python -m src.train_resnet
```

Trains for 15 epochs on CIFAR-10 subset, logs metrics to `experiments/training_log.json`.

## Project Structure

- `src/` — Core implementation (blocks, model, training)
- `experiments/` — Data and training outputs
- `notebooks/` — Analysis and visualization
