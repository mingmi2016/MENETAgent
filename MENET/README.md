# MENET Algorithm Core

MeNet is a mixed-effects deep neural network architecture for multi-environment agronomic traits prediction. This directory contains model deployment, training steps, prediction steps, and model parameter settings.

This is the algorithm boundary of the MENET Agent project. It contains no HTTP API, user accounts, conversations, database, uploads, or web interface.

## Contents

```text
MENET/
├── network/                         # Neural network definitions
├── utils/                           # Data, loss, training, IG and relatedness utilities
├── configs/                         # Algorithm parameter templates
├── data/                            # Original format example
├── menet.py                         # MENET training entry point
└── train_trait_specific_encoder.py  # Trait-specific encoder entry point
```

## Standalone Use

```bash
cd /home/mingmi/workspace/MenetAgent/MENET
../.venv/bin/python train_trait_specific_encoder.py -p culmlength -d cuda
../.venv/bin/python menet.py -p culmlength -d cuda
```

Input data follows the original layout:

```text
data/
├── genotype/genotype.csv
├── phenotype/{trait}.csv
└── split/
    ├── train_index.txt
    ├── valid_index.txt
    └── test_index.txt
```

The Agent application calls this core through an explicit Python integration boundary in `../agent_app/agent/tools.py`.
