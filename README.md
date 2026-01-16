# FedGANBLR

Federated tabular data generator and benchmark for FedGANBLR . This repo runs centralized vs. federated experiments across tabular datasets and compares performance.

## Features
- Run centralized and federated training comparisons
- Cross-validation and repeated experiments
- Configurable datasets, federated parameters, and training hyperparameters
- Scriptable experiments via `main.py`

## Requirements
- Python 3.8+
- Typical ML libraries (e.g., numpy, pandas, scikit-learn, PyTorch or TensorFlow depending on implementation)
- See `requirements.txt` 

## Installation
1. Clone the repo:
   git clone <repo-url>
2. Create a venv and activate:
   python3 -m venv .venv
   source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
   (or install project-specific packages manually)

## Usage
Edit dataset list and configuration in `main.py`, then run:
python3 main.py

`main.py` calls:
- `compare_real_central_fed_cv_all_datasets(DATASETS, **compare_cfg)`

Key configuration (`compare_cfg`) includes:
- n_splits, n_repeats, random_state — CV settings
- num_clients, num_rounds, local_epochs — federated settings
- epochs_ganblr, batch_ganblr, warmup_ganblr — GAN/BLR training
- dir_alpha, gamma, alpha_dir, cpt_mix — algorithm hyperparameters

Populate `DATASETS` with dicts:
- name — dataset name
- alt — list of alternate names
- data_id — OpenML id (optional)
- target — target column name
- ef_bins — effective bin parameter (optional)

## Project layout (top-level)
- main.py — entry point for experiments
- evaluation.py — evaluation and experiment orchestration (imported by main)
- other modules — model, data loaders, utils (see repo for details)

## Adding datasets
Add dataset entries to `DATASETS` in `main.py`. Ensure the dataset loader in the repo supports the dataset or add a loader mapping by name/data_id.

## Outputs
Experiment results, logs, and any saved models are produced by the evaluation routines. Check configured output directories in the evaluation code.

## Contributing
- Open issues for bugs/features
- Send PRs with tests and concise commits

## License
Specify project license in `LICENSE` file (add one if missing).
