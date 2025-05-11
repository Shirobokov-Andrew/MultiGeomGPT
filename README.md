# MultiGeometry Transformer

A PyTorch implementation of transformers that operate in different geometric spaces (Euclidean, Hyperbolic, and Spherical) for language modeling tasks.

## Overview

This repository contains the code for experimenting with transformers that leverage different geometries for attention mechanisms. The implementation includes:

- Multi-geometry transformer blocks that can use Euclidean, Hyperbolic, and Spherical spaces within different attention heads
- Fully hyperbolic transformer blocks with operations happen in Lorentz model
- Joint geometry blocks that specialize in a specific selected geometry
- Customizable configuration for layer distribution across geometries
- Training pipeline with Pytorch Distributed Data Parallel (DDP) support

This work explores the hypothesis that different geometric spaces may be better suited for capturing different types of relationships in language data.

## Installation

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Shirobokov-Andrew/MultiGeomGPT.git
cd MultiGeomGPT
```

2. Set up the conda environment:
```bash
conda env create -f env.yaml
conda activate env_name
```

## Project Structure
```
├── config/                 # Configuration settings
│   ├── model_config.py     # Model architecture config
│   └── train_config.py     # Training hyperparameters
├── data/                   # Data preparation and loading scripts
│   ├── wikitext103/        # WikiText-103 dataset
│   ├── wikitext2/          # WikiText-2 dataset
│   └── shakespeare_char/   # Character-level Shakespeare dataset
├── model/                  # Model implementation
│   ├── model.py            # Main model architecture
│   └── lmath.py            # Math operations for hyperbolic space
├── utils/                  # Utility functions
│   ├── loader.py           # Distributed data loader
│   ├── logging.py          # Logging utilities
│   ├── muon.py             # Muon optimizer implementation
│   └── optimization.py     # Optimization utilities
├── train.py                # Main training script
└── run_exp.sh              # Script to run distributed training
```
## Key Features

### Geometry-Specific Implementations

- Euclidean Geometry: Standard transformer operations in Euclidean space
- Hyperbolic Geometry: Operations in the Lorentz model of hyperbolic space
- Spherical Geometry: Operations on the unit sphere

### Multi-Geometry Options

The model supports several configurations:

1. Multi-Geometry Block: Combines different geometries within a single transformer block
2. Joint Geometry Blocks: Each block specializes in a specific geometry
3. Fully Hyperbolic Blocks: All operations performed in hyperbolic space

### Training Configuration

The training pipeline supports:

- Distributed training with PyTorch DDP
- Automatic Mixed Precision training with bfloat16
- Different optimizers including Muon, Prodigy, Adam
- Configurable learning rates for curvature parameters
- Tensorboard logging and checkpointing

## Usage

### Data Preparation

The repository supports several datasets:

- WikiText-103
- WikiText-2 
- Shakespeare (character-level)
- FineWeb

To prepare the WikiText-103 dataset:
```bash
cd data/wikitext103
python prepare.py
```

### Training

To train the model:
```bash
# For single GPU training
python train.py
```

```bash
# For distributed training
bash run_exp.sh
```

### Configuration

The model and training can be configured by modifying:

- `config/model_config.py` for model architecture
- `config/train_config.py` for training hyperparameters

Key configuration options:
```python
# Model architecture configuration
model_config = MultiGeomGPTConfig(
    n_layers=6,                # Number of transformer layers
    n_embd=256,                # Embedding dimension  
    n_heads=2,                 # Number of attention heads
    head_dim=128,               # Dimension per head
    multi_geom_block=False,    # Use multi-geometry blocks
    layers_order=("sph", "hyp", "euc"),  # Order of geometry layers
    n_euc_layers=2,            # Number of Euclidean layers
    n_hyp_layers=2,            # Number of hyperbolic layers  
    n_sph_layers=2,            # Number of spherical layers
    lm_head_mode='euc',        # LM head geometry
    curvature=1.0,             # Initial hyperbolic curvature
)
```
```python
# Training configuration
train_config = TrainConfig(
    device_batch_size=16,      # Batch size per GPU
    global_batch_size=8 * 16,  # Global batch size
    context_length=512,        # Sequence length
    num_iterations=10000,      # Number of training iterations
)
```
## Model Variants

The model supports several variants:

1. *Euclidean*: Standard transformer implementation (`n_euc_layers` > 0)
2. *Hyperbolic*: Layers operating in hyperbolic space (`n_hyp_layers` > 0)
3. *Spherical*: Layers operating on the unit sphere (`n_sph_layers` > 0)
4. *Mixed*: A combination of different geometries in sequential order

## Monitoring

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir=runs/
```
The training script logs:
- Training and validation loss
- Gradient norms
- Hyperbolic curvature parameters
- Generated text samples

## Acknowledgements

This implementation is inspired by and builds upon research in geometry-aware neural networks, particularly in hyperbolic and spherical spaces. We acknowledge the contributions of previous works in this field:\
[`nanoGPT`](https://github.com/kellerjordan/nanoGPT) \
[`HyperbolicCV`](https://github.com/kschwethelm/HyperbolicCV) \
[`nGPT`](https://github.com/NVIDIA/ngpt)
