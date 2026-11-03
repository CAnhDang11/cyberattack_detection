# UNSW-NB15: 10-class centralized, DDP and federated experiments

The active target is `attack_cat`: **Normal plus nine attack categories**. The binary `label` column is excluded from both inputs and targets. Start with [the complete techniques and run guide](docs/TECHNIQUES_AND_RUNS.md). Generate comparison reports from your completed runs using `python -m unsw.compare PATH_TO_RUNS`.

## Setup and data

Run from the repository root, using Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-flower.txt  # optional Flower
python -m unsw.run prepare --output experiments/data
```

For a quick run, add `--limit-per-class 200` to `prepare`. This caps training data per class, keeps the supplied test CSV unchanged, and is not a representative full-data benchmark. Preparation overwrites that output directory's artifacts; use distinct directories for distinct datasets.

Mặc định dùng toàn bộ **175.341 dòng training CSV**, không giới hạn mẫu mỗi lớp. Với seed 42, dữ liệu được tách thành **140.273 dòng học + 35.068 dòng validation**; tập test riêng có **82.332 dòng**. Mỗi lệnh train in số mẫu thực tế để kiểm tra. Chạy đầy đủ:

```bash
python -m unsw.run prepare --output experiments/data --limit-per-class 0
python -m unsw.run centralized --epochs 10 --output experiments/centralized_full
```

## Exact training commands

```bash
# Shared PyTorch MLP centralized reference
python -m unsw.run centralized --epochs 10 --output experiments/centralized
# Adapted notebook model families: logistic, mlp, rf, xgboost
python -m unsw.run baseline --baseline-model logistic --epochs 30 --output experiments/logistic
python -m unsw.run baseline --baseline-model mlp --epochs 30 --output experiments/sklearn_mlp
python -m unsw.run baseline --baseline-model rf --output experiments/rf
# pip install xgboost before using this optional baseline
python -m unsw.run baseline --baseline-model xgboost --output experiments/xgboost
# CPU DDP, two real workers (requires loopback sockets)
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 -m unsw.run ddp --epochs 10 --output experiments/ddp
# One-process CPU fallback
python -m unsw.run ddp --epochs 1 --output experiments/ddp_smoke
# Flower local clients, using Flower's actual strategy aggregation
python -m unsw.run fedavg --clients 4 --case equal_iid --rounds 10 --local-epochs 1 --participation 1 --output experiments/fedavg
python -m unsw.run fedprox --clients 4 --case unequal_non_iid --alpha 0.5 --mu 0.01 --rounds 10 --local-epochs 1 --participation 0.75 --output experiments/fedprox
# All four cases and repeated seeds; DDP here is the one-process fallback
python -m unsw.suite --seeds 42 43 44 --epochs 10 --rounds 10 --clients 4 --alpha 0.5
python -m unsw.compare experiments
python -m unittest discover -s tests -v
# Alternatively, when pytest is installed:
python -m pytest tests -q
```

Use `--data PATH/data.npz` for another prepared dataset, `--seed`, `--batch-size`, `--lr`, `--hidden`, `--hidden-layers`, and `--case` on training commands. Use `--device cuda` for centralized/DDP GPU runs. Flower's local runner currently uses CPU. All four cases are `equal_iid`, `unequal_iid`, `equal_non_iid`, `unequal_non_iid`.

## Compare centralized model families

All `baseline` models train centrally on the full fit split, just like `centralized` (the PyTorch MLP). Prepare once and reuse the same data file for every model and training seed:

```bash
python -m unsw.run prepare --seed 42 --limit-per-class 0 --output experiments/data
python -m unsw.centralized_suite --data experiments/data/data.npz \
  --output experiments/aligned_v1 --seeds 42 43 44 \
  --epochs 30 --batch-size 256 --hidden 512 --hidden-layers 4 --lr 0.001 --l2 0 \
  --n-estimators 500 --max-depth 10 --tree-lr 0.1
```

The suite runs torch_mlp, logistic, sklearn mlp, rf and xgboost in separate processes and writes `comparison.csv` and `REPORT.md`. Install optional XGBoost first (`python -m pip install xgboost`), or select a subset with `--models torch_mlp logistic mlp rf`. Use a fresh output directory for each parameter configuration.

| Control | Applies to |
| --- | --- |
| `--epochs` | PyTorch MLP, sklearn MLP, SGD logistic (passes over fit data) |
| `--lr` | PyTorch/sklearn MLP learning rate; SGD logistic constant step size |
| `--hidden` | Both MLPs start at width 512 by default, halving each layer (minimum 2) with ReLU |
| `--hidden-layers` | Both MLPs; default 4 gives `(512, 256, 128, 64)` |
| `--batch-size` | Both MLPs; SGD logistic updates per sample |
| `--l2` | MLP/SGD regularization; defaults to zero; library scaling differs |
| `--n-estimators` | RF tree count; XGBoost boosting rounds; default 500 |
| `--max-depth` | Both tree models; omitted defaults are RF 25, XGBoost 10 |
| `--tree-lr` | XGBoost shrinkage only |

New runs use larger defaults: four hidden layers for both MLPs, 500 RF trees with depth 25, and 500 XGBoost rounds with depth 10. These increase training time, memory, and federated communication; validate macro F1 before assuming an improvement. Logistic regression remains linear. To reproduce the former architecture, pass `--hidden 128 --hidden-layers 2`; for former tree budgets, pass `--n-estimators 100 --max-depth 15` (RF) or `--n-estimators 100 --max-depth 8` (XGBoost). Existing checkpoints require their original architecture when loading.

Both MLPs now use Adam; sklearn MLP runs the requested epoch budget without early stopping. PyTorch evaluates each epoch and restores the best validation macro F1 checkpoint; classical estimators use their final fit. Equal controls do not imply equal computation or identical numerical training. `--lr` now replaces the former hardcoded 0.05 for SGD and sklearn MLP; sklearn MLP's former optimizer was SGD. New baseline runs save actual library settings in `estimator_params.json`, and results identify the model family.

Compare `selected_val_macro_f1` across configurations and seeds to choose a model. Reserve test accuracy and test macro F1 for final reporting; repeatedly selecting parameters using test scores biases the evaluation. Check `per_class.csv` for rare classes. Training seeds reuse the prepared split; changing the preparation seed creates a different split.

Each run appends to `training.log` in its `--output` directory, including configuration, client distributions, epoch/round metrics, final test metrics, per-class scores and confusion matrix. A timestamped `NEW RUN` separator marks each run; existing log content is preserved. DDP writes this log on rank zero only. Failed training inside the runner records its traceback. Other artifacts (checkpoints, CSV and JSON files) are still replaced when reusing an output directory; use distinct output directories to preserve those artifacts too.

```bash
cat experiments/centralized_full/training.log
tail -f experiments/centralized_full/training.log
# Create missing logs for already completed runs from their saved results:
python -m unsw.logging_utils experiments
```

Historical logs generated this way are marked `RECONSTRUCTED`; existing logs are preserved.

## Project layout

- `unsw/data.py`, `labels.json`: shared preprocessing, encoding, partitions and audit.
- `unsw/model.py`: one 10-logit MLP and cross-entropy training implementation.
- `unsw/run.py`, `baseline.py`: centralized/DDP entry point and adapted notebook families.
- `unsw/federated.py`: Flower NumPyClient, FedAvg and FedProx local orchestration.
- `unsw/evaluation.py`, `compare.py`, `suite.py`: metrics, reporting and repeatable experiment grid.
- `tests/`: small synthetic tests and CPU smoke runs.
- `experiments/`: multiclass prepared data, runs, reports and validation evidence.
- `archive/`: preserved binary DDP and custom-socket FL starter trees, including their logs/checkpoints.
- Original CSVs, `CSV Files/`, notebooks, `output_preprocess/`, and `output_baseline/` are preserved. Their saved results are historical binary results.

No code was declared dead or deleted. The reviewed starter modules are reachable through their old entry points; they were moved intact to `archive/` because the shared multiclass implementation supersedes their binary pipelines. The original README is saved in `docs/BINARY_BASELINE_README.md`. Historical notebooks have an explicit binary-experiment notice.
