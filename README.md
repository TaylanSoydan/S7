# S7: Selective and Structured State Spaces for Event-Based Data

A JAX/Flax implementation of S7, a state space model designed for event-based and irregularly-sampled sequence data. S7 extends the S5 architecture with event-aware discretization, enabling effective processing of event streams from neuromorphic sensors, time series, and long-range sequence benchmarks.

**Paper:** [S7: Selective and Structured State Spaces for Event-Based Data](https://arxiv.org/abs/2410.03464)

## Installation

```bash
conda create --name s7 python=3.9
conda activate s7
pip install -r requirements.txt
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. All configs are in `configs/`.

**Set your data directory** in `configs/system/local.yaml`:
```yaml
data_dir: /path/to/your/data
```

## Usage

### Training

```bash
# Train on Spiking Speech Commands (default)
python run_training.py

# Train on a specific task
python run_training.py task=spiking-heidelberg-digits
python run_training.py task=dvs-gesture
python run_training.py task=text

# Override config values
python run_training.py task=listops training.per_device_batch_size=32
```

### Evaluation

```bash
python run_evaluation.py task=text checkpoint=./best_text/checkpoints
```

Ensure the model and task config parameters match the checkpoint (e.g., if the checkpoint was trained with `d_ssm=70`, the config must also use `d_ssm=70`).

### Supported Tasks

| Category | Tasks |
|---|---|
| Event Streams | `spiking-heidelberg-digits`, `spiking-speech-commands`, `dvs-gesture` |
| Long Range Arena | `listops`, `text` (IMDB), `retrieval` (AAN), `image` (CIFAR), `pathfinder`, `pathx` |
| Time Series | `eigenworms`, `personactivity`, `walker` |
| Language Modeling | `ptb`, `wikitext2` |

## Project Structure

```
S7/
├── run_training.py          # Training entry point
├── run_evaluation.py        # Evaluation entry point
├── configs/                 # Hydra configuration files
│   ├── base.yaml            # Default config
│   ├── task/                # Task-specific configs
│   ├── model/               # Model architecture configs
│   ├── system/              # System/path configs
│   └── logging/             # Logging configs
├── src/                     # Core library
│   ├── ssm.py               # S7 state space model
│   ├── ssm_init.py          # HiPPO initialization
│   ├── layers.py            # SSM layers and pooling
│   ├── seq_model.py         # Full sequence models (classification, retrieval)
│   ├── train_utils.py       # Training/evaluation step functions
│   ├── trainer.py           # Training loop with logging
│   ├── transform.py         # Data augmentations (CutMix, jitter, etc.)
│   ├── s5_compat.py         # Base dataset classes
│   └── dataloaders/         # Dataset implementations
│       ├── event_streams.py # SHD, SSC, DVS Gesture
│       ├── lra.py           # Long Range Arena benchmarks
│       ├── timeseries.py    # EigenWorms, PersonActivity, Walker
│       ├── text.py          # PTB, WikiText language modeling
│       └── collate.py       # Collate functions and dataloader utilities
└── requirements.txt
```

## Acknowledgements

This codebase builds on the following projects:

- [event-ssm](https://github.com/Efficient-Scalable-Machine-Learning/event-ssm) — event-based SSM code for event stream processing
- [S5](https://github.com/lindermanlab/S5) — S5 model and Long Range Arena benchmark integration
- [ode-lstms](https://github.com/mlech26l/ode-lstms) — physics benchmark datasets (PersonActivity, Walker2d)
