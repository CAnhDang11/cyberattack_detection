# Techniques, experimental protocol and run reference

## Target and encoding

Inspection of the actual CSVs found exactly the following mapping, saved once in `unsw/labels.json`:

| Index | Category |
|---|---|
| 0 | Normal |
| 1 | Analysis |
| 2 | Backdoor |
| 3 | DoS |
| 4 | Exploits |
| 5 | Fuzzers |
| 6 | Generic |
| 7 | Reconnaissance |
| 8 | Shellcode |
| 9 | Worms |

Encoding is exact and case sensitive. Missing labels, whitespace variants, and unknown spellings raise an error; they are never inferred from binary `label`. Decoding uses this same order. Prepared datasets, results and checkpoints include the mapping. Loading prepared data checks mapping equality. Old binary NPZs must not be passed to the new trainer.

## Split and preprocessing

The supplied training/testing CSV separation is preserved. A five-fold StratifiedGroupKFold split selects one validation fold from training only. Raw feature hashes group duplicate training rows so identical features do not cross fit/validation. Test labels influence neither partitions nor model selection. Group stratification is approximate, especially for rare categories. Labels are validated before fitting.

Preprocessing excludes `id`, `label`, and `attack_cat`, renames `ackdat` to `ackdata`, applies log1p to nonnegative numeric features, standardizes using fit-subset mean/standard deviation, and one-hot encodes categorical features, dropping the first category. Numeric missing values use fit-subset medians (all-missing columns use zero). Categorical missing values use a sentinel; unseen categories are all-zero, like the reference category. Negative/infinite numeric values fail clearly. No PCA is used. Statistics and vocabulary are fitted only on the training fit subset, not validation/test. The prepared artifacts are a shared offline preprocessing stage; the local FL simulator is not a privacy-preserving preprocessing deployment.

The original notebooks applied forward/backward filling and binary threshold selection. Their four model families are retained in multiclass CLI form, but binary thresholds, binary FPR constraints and binary confidence intervals do not transfer to this task. Automatic hyperparameter search is not implemented in the new runner. Choose settings using validation results only; do not tune against the reported test metrics.

The original split has feature duplicates across CSVs. `audit.json` reports their count. Test row identities never enter a training partition, but feature-level overlap already present in the source data is retained to respect the intended split. Therefore an assertion of zero feature overlap would be false. This limits generalization claims; an additional unseen-feature test analysis can be conducted separately.

## Four client partition cases

Partitions use only fit-subset row indices. Equal capacities differ by at most one sample. Unequal capacities follow deterministic weights 1, 2, ..., K with at least one sample per client; largest remainders allocate integer leftovers. Seeds control permutation and allocation.

IID allocation assigns randomly interleaved samples with probabilities proportional to remaining capacity. Class proportions approximate the global distribution, subject to sampling noise. Non-IID allocation first draws class-specific Dirichlet client preferences with configurable `alpha`, then multiplies them by remaining capacity during allocation. Smaller alpha usually increases skew. This is a capacity-constrained Dirichlet scheme, not unconstrained Dirichlet class splitting. It guarantees exact sizes and full coverage, at the cost of moderating skew as capacities fill. Rare classes may be absent from individual clients. Each sample appears in exactly one partition. `clients.csv` records all ten class counts and sample counts; `partitions.npz` saves exact row indices. Changing seed or client count changes partitions; all methods given identical settings receive the same partition scheme.

## Models and optimization

The common PyTorch MLP is input → 512 → 256 → 128 → 64 → 10 by default (`--hidden 512 --hidden-layers 4`; widths halve with a minimum of 2), with ReLU between hidden layers. It returns logits. CrossEntropyLoss accepts integer class indices and internally applies log-softmax. Prediction uses softmax and argmax, then the saved mapping. Adam and ordinary unweighted cross entropy are shared across centralized, DDP and Flower clients. No class weights or oversampling are silently applied. The adapted sklearn SGD-logistic, Adam-MLP, random forest and optional XGBoost baselines preserve the notebook model families with expanded capacity defaults, but are distinct algorithms from the common PyTorch MLP.

Centralized training pools every client partition. Partition cases therefore do not change available examples, although order can change stochastic training. Classical baselines fit once and report terminal fit/validation loss; epoch histories are provided for PyTorch training, not fabricated for sklearn tree fits.

DDP wraps the common model, synchronizes gradients during backpropagation, uses DistributedSampler and calls set_epoch every epoch. Training loss sums and counts are all-reduced across ranks. Rank zero evaluates the complete validation/test set without distributed padding, so every held-out row is counted once and its metrics are global. Only rank zero writes metrics/checkpoints. Validation macro F1 selects the checkpoint, and test is evaluated once after selection. NCCL is used for CUDA and Gloo for CPU. DistributedSampler may pad training by fewer than world-size examples per epoch to keep synchronized batch counts; this does not change the saved disjoint client partitions. DDP uses global shuffled sampling across the partition union; these cases do not model client-local DDP rank skew. For single-worker mode there is no communication or gradient synchronization.

Flower runs separate local client objects sequentially in one process without Ray or a network server. Each client receives model arrays, trains on its own indices with a fresh local Adam optimizer each round, and returns arrays/sample counts/scalar training loss. Flower's FedAvg or FedProx strategy performs weighted aggregation through FitRes. The server aggregation code receives no raw examples. This is a local experiment harness using Flower, not a multi-host Flower deployment. Participation selects ceil(fraction × client_count), uniformly without replacement using the seed. FedAvg weights participating updates by local sample count. FedProx adds mu/2 × ||local − round-start-global||² during local optimization; reported train loss is predictive cross entropy excluding this penalty. Every round evaluates validation, and the best validation macro-F1 state is tested.

The server-side strategies use Flower's legacy `flwr.server.strategy` API (requirements permit Flower 1.x). See the [official FedProx API](https://flower.ai/docs/framework/1.22/en/ref-api/flwr.server.strategy.FedProx.html). The implementation must be smoke-tested against an installed Flower version before treating its results as validated.

## Evaluation and artifacts

Each completed run saves config.json, labels.json, partitions.npz, clients.csv, history.csv, results.json, results.csv, per_class.csv and confusion_matrix.csv. Neural checkpoints save dimensions, configuration, mapping and selected model state; classical checkpoints are joblib dictionaries containing model/mapping/config. Use only trusted checkpoint files. Confusion-matrix rows are true labels, columns predicted labels, both in mapping order. Per-class support is recorded even when no predictions select a class.

Accuracy and weighted precision/recall/F1 emphasize common classes. Macro precision/recall/F1 weight every class equally and reveal minority failures. Undefined precision/recall use zero. ROC-AUC is macro one-vs-rest using all ten probability columns; missing test classes yield null with an explicit reason. Evaluation loss is multiclass log loss. Training histories use sample-weighted predictive cross entropy. Training time includes validation and orchestration overhead; evaluation time includes final inference and metrics. Final train_loss is the last training step's loss, while test metrics use the validation-selected checkpoint.

Flower communication volume counts uncompressed model-array upload plus download for participating clients, cumulatively. It excludes serialization/protocol bytes and is an estimate of payload, not measured wire traffic. DDP communication is null because the implementation does not profile collectives. Centralized communication is also null. Compare elapsed times only under matched hardware/work budgets. An epoch over the centralized union and one federated round are not equivalent compute budgets.

Normal/Generic/Exploits dominate the full dataset. Worms has only 130 train and 44 test examples. Minority per-class scores have high uncertainty. The quick per-class training cap deliberately changes the training prior, so quick results cannot establish full-data superiority. Run full training, multiple seeds and validation-only tuning before scientific claims.

## Running and reproducing

All exact commands are in the root README. For full results run preparation without a cap, then `python -m unsw.suite --seeds 42 43 44`. Choose unique output directories. Run two-process DDP separately using torch.distributed.run; suite's DDP default is a one-process CPU fallback. Use matching prepared data and seeds for comparisons. `python -m unsw.compare experiments` collects only completed results, never substitutes numbers for failed/missing methods. Inspect configs to distinguish training caps, epochs and DDP world size. Keep comparisons from different prepared datasets in separate experiment roots.

Tests use unittest (also discoverable by pytest), synthetic tensors and deterministic partitions. Optional Flower tests skip only when its import is absent. The test log and environment limitations are recorded in experiments/VALIDATION.md. This environment cannot install Flower/pytest because package-index DNS/network access is unavailable. Full Flower execution and multi-host deployment are consequently unverified; no corresponding benchmark results are claimed.
