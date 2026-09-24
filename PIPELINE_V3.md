
# Metadata-Guided Test-Time Adaptation — Pipeline V3

## 1. Overview

This repository implements an experimental pipeline for **metadata-guided Test-Time Adaptation (TTA)** under temporal distribution shift.

The central idea is to exploit metadata that is available at test time as an auxiliary supervised signal for adapting a model **without using the labels of the main prediction task**.

The pipeline supports:

- temporal source training;
- model hyperparameter tuning;
- auxiliary-head hyperparameter tuning;
- metadata-guided TTA;
- episodic and cumulative adaptation;
- annual resets;
- drift-triggered resets using ADWIN;
- Temporal Gradient EMA adaptation;
- TENT;
- an optional supervised OOD oracle/reference;
- independent experiment and split seeds;
- single-run and multi-run execution;
- explicit safeguards against test-set leakage.

The V3 pipeline has been exercised successfully with an end-to-end smoke test covering:

1. dataset loading;
2. temporal splitting;
3. source-model tuning;
4. auxiliary-head tuning;
5. TTA tuning;
6. hyperparameter selection;
7. final retraining from scratch;
8. ID evaluation;
9. real-OOD evaluation;
10. supervised reference evaluation;
11. result serialization.

CoTTA is currently reserved as a future method and is **not implemented**.

---

# 2. Research Goal

Let an input sample be:

\[
x_t
\]

with:

- main target \(y_t\);
- auxiliary metadata target \(m_t\).

During normal supervised source training, the main target is available.

During Test-Time Adaptation, however:

\[
y_t \quad \text{is NOT available to the adaptation algorithm}
\]

while:

\[
m_t \quad \text{is available}
\]

The objective is therefore to use the metadata target \(m_t\) to adapt the representation of \(x_t\), with the goal of improving prediction of \(y_t\) under temporal distribution shift.

Conceptually:

```text
                     SOURCE TRAINING

x ──> SSL embedding ──> Adapter ──> Shared features ──> Main Head
                                                        │
                                                        ▼
                                                   main target


                     TEST-TIME ADAPTATION

x_t ──> SSL embedding ──> Adapter ──> Shared features ──> Aux Head
                             ▲                            │
                             │                            ▼
                             └──── metadata loss <── aux_target

After adaptation:

x_t ──> SSL embedding ──> UPDATED Adapter
                             │
                             ▼
                        Shared features
                             │
                             ▼
                          Main Head
                             │
                             ▼
                      main prediction
```

The main-task label is used only for evaluation after the prediction has been produced.

---

# 3. Repository Structure

The main V3 structure is:

```text
metadata_tta_v3/
│
├── configs/
│   ├── batches/
│   ├── datasets/
│   ├── experiments/
│   └── tuning/
│
├── scripts/
│   ├── run.py
│   └── run_batch.py
│
├── src/
│   └── metadata_tta/
│       ├── cache/
│       ├── data/
│       ├── evaluation/
│       ├── experiment/
│       ├── models/
│       ├── protocols/
│       ├── results/
│       ├── training/
│       ├── tta/
│       ├── tuning/
│       ├── utils/
│       ├── config.py
│       └── reproducibility.py
│
├── environment.yml
├── requirements-lock.txt
└── PIPELINE_V3.md
```

Important modules include:

```text
src/metadata_tta/protocols/v3.py
```

for temporal train/validation/test construction;

```text
src/metadata_tta/tuning/v3.py
```

for model, auxiliary-head, and TTA tuning;

```text
src/metadata_tta/experiment/runner.py
```

for final experiment orchestration;

```text
src/metadata_tta/evaluation/evaluator.py
```

for TTA stream evaluation;

and:

```text
src/metadata_tta/tta/
```

for the individual adaptation algorithms.

---

# 4. Environment Setup

## 4.1 Requirements

The project uses:

- Python 3.11;
- PyTorch;
- NumPy;
- pandas;
- scikit-learn;
- PyYAML;
- CapyMOA;
- ADWIN through CapyMOA;
- supporting scientific Python libraries.

Exact dependency versions are stored in:

```text
requirements-lock.txt
```

The environment is defined by:

```text
environment.yml
```

## 4.2 Create the Conda environment

From the repository root:

```bash
conda env create -f environment.yml
```

Activate it with:

```bash
conda activate metadata_tta
```

## 4.3 Verify the environment

For example:

```bash
python --version
```

The expected Python version is:

```text
Python 3.11
```

Useful dependency checks:

```bash
python - <<'PY'
import torch
import numpy
import pandas
import sklearn
import yaml
import capymoa

print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)

print("[CHECK][PASS] core dependencies import correctly")
PY
```

## 4.4 PYTHONPATH

The project currently uses the `src/` layout without requiring package installation.

For direct module checks use:

```bash
PYTHONPATH=src python ...
```

The main scripts add/use the project structure appropriately, so normal experiment execution is performed from the repository root.

Example:

```bash
python scripts/run.py --help
```

## 4.5 Recreating an existing environment

The exact pinned Python dependencies are listed in:

```text
requirements-lock.txt
```

They include, among others:

```text
capymoa==0.11.0
numpy==1.26.4
pandas==2.2.2
PyYAML==6.0.3
scikit-learn==1.7.2
torch==2.2.2
```

The complete list should always be taken directly from `requirements-lock.txt` rather than duplicated manually.

---

# 5. Dataset Format

The pipeline operates on precomputed embeddings.

The SSL representation is therefore considered fixed: the TTA pipeline does not fine-tune the original SSL encoder.

Each dataset contains:

```text
feat_0
feat_1
...
feat_D
target
aux_target
```

where:

- `feat_*` are the fixed embedding dimensions;
- `target` is the main-task class;
- `aux_target` is the metadata class.

For example, the Yearbook representation contains 2048 embedding features:

```text
feat_0 ... feat_2047
target
aux_target
```

The exact physical organization of files is handled by the data-loading layer.

---

# 6. Dataset Configurations

Dataset-specific information is stored under:

```text
configs/datasets/
```

Each configuration defines:

- dataset name;
- data root;
- number of main classes;
- number of auxiliary classes;
- target column;
- auxiliary-target column;
- split proportions;
- temporal protocol.

## 6.1 FMoW

FMoW uses:

```text
main classes:       52
auxiliary classes:  11
years:               2002–2017
```

Protocol:

```text
pseudo-source:  2002–2008
pseudo-OOD:     2009–2012
final source:   2002–2012
real OOD:       2013–2017
```

## 6.2 HuffPost

HuffPost uses:

```text
main classes:       11
auxiliary classes:   5
years:               2012–2018
```

Protocol:

```text
pseudo-source:  2012–2014
pseudo-OOD:     2015
final source:   2012–2015
real OOD:       2016–2018
```

## 6.3 Yearbook

Yearbook uses:

```text
main classes:        2
auxiliary classes:  23
years:               1930–2013
```

The auxiliary target represents the **U.S. state of the college**.

Protocol:

```text
pseudo-source:  1930–1966
pseudo-OOD:     1967–1970
final source:   1930–1970
real OOD:       1971–2013
```

---

# 7. Per-Year 70/10/20 Split

Every year is independently divided into:

```text
70% training
10% validation
20% test
```

Internally, this is represented by:

```text
TRAIN
VALIDATION
TEST
```

The three sets must be mutually disjoint.

Conceptually:

```text
                    YEAR t
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
      TRAIN       VALIDATION       TEST
       70%            10%           20%
```

The split is determined by `split_seed`.

The same `split_seed` must reproduce exactly the same split membership.

---

# 8. Temporal Experimental Protocol

The V3 protocol separates four temporal regions.

## 8.1 Pseudo-source

The pseudo-source period is used for **model hyperparameter tuning**.

Example for FMoW:

```text
2002 ───────────────────── 2008
            pseudo-source
```

## 8.2 Pseudo-OOD

Pseudo-OOD years simulate future temporal distribution shift.

They are used for **TTA hyperparameter tuning**.

For FMoW:

```text
2009 ───────────────────── 2012
             pseudo-OOD
```

Critically, TTA tuning uses only the:

```text
10% VALIDATION split
```

of each pseudo-OOD year.

The 20% test split is not used.

## 8.3 Final source

After all hyperparameters have been selected, the final model is trained **from scratch** using the complete source period.

For FMoW:

```text
2002 ───────────────────── 2012
             final source
```

This is exactly the union of:

```text
pseudo-source + pseudo-OOD
```

## 8.4 Real OOD

The real OOD years are reserved for final evaluation.

For FMoW:

```text
2013 ───────────────────── 2017
              real OOD
```

Final OOD accuracy is computed on the:

```text
20% TEST split
```

of each real-OOD year.

---

# 9. Leakage Prevention

A fundamental V3 design requirement is that the test set never participates in hyperparameter selection.

The intended information flow is:

```text
PSEUDO-SOURCE
70% train ──────> model training
10% val   ──────> model HP selection
20% test  ──────> NOT USED FOR TUNING


PSEUDO-OOD
10% val   ──────> TTA stream for HP selection
20% test  ──────> NOT USED FOR TUNING


REAL OOD
20% test  ──────> final reported evaluation
```

The tuner is designed so that it does not need the pseudo-OOD test loaders.

Violations of important protocol assumptions should fail rather than silently continue.

---

# 10. Model Architecture

## 10.1 Fixed SSL embedding

The input is a precomputed SSL embedding:

\[
z = f_{\mathrm{SSL}}(x)
\]

The SSL encoder itself is not updated.

## 10.2 Residual Adapter

A residual adapter transforms the embedding before classification.

Conceptually:

\[
z' = z + A(z)
\]

The adapter is the component adapted by metadata-guided TTA.

## 10.3 Shared Feature Block

The shared feature block contains the common representation used by the heads.

Its architecture includes:

```text
Linear
BatchNorm
ReLU
Dropout
```

## 10.4 Main Head

The Main Head predicts:

\[
p(y \mid x)
\]

and is trained using the main supervised classification objective during source training.

---

# 11. Single Head Source Model

The source model initially contains:

```text
SSL embedding
      │
      ▼
Residual Adapter
      │
      ▼
Shared Feature Block
      │
      ▼
Main Head
```

During source training:

```text
Adapter       TRAINABLE
Shared block  TRAINABLE
Main Head     TRAINABLE
SSL embedding FIXED
```

The objective is main-task cross entropy.

---

# 12. Double Head Model

After source training, a Double Head model is created.

The trained main path is copied exactly:

```text
Single Head

Adapter ── Shared ── Main


              exact copy
                  │
                  ▼


Double Head

Adapter ── Shared ── Main
             │
             └──────── Aux
```

Before auxiliary-head training, the main predictions of the Single Head and Double Head models must be identical.

This equivalence is explicitly checked.

---

# 13. Auxiliary Head Training

After copying the trained Single Head model:

```text
Adapter       FROZEN
Shared block  FROZEN
Main Head     FROZEN
Aux Head      TRAINABLE
```

The Aux Head is trained to predict metadata.

For auxiliary-head training:

```text
train = concatenation of the 70% source splits
val   = concatenation of the 10% source splits
```

The auxiliary loss is cross entropy.

The auxiliary learning rate is tuned independently before TTA hyperparameter tuning.

---

# 14. Model Hyperparameter Tuning

Model tuning is performed on pseudo-source years.

Candidate configurations are explicitly enumerated rather than generated through a large Cartesian product.

The tuned quantities include:

```text
learning rate
temporal learning rate
weight decay
shared hidden dimension
```

For every candidate:

1. create a fresh model;
2. train sequentially over pseudo-source years;
3. evaluate validation accuracy for each pseudo-source year;
4. compute mean validation accuracy;
5. select the candidate with the highest mean validation accuracy.

The 20% test split does not participate.

---

# 15. Auxiliary-Head Hyperparameter Tuning

After selecting the source-model configuration:

1. train the pseudo-source model using the selected model HPs;
2. create the Double Head model;
3. copy the Single Head main path exactly;
4. tune the Aux Head learning rate;
5. select the auxiliary configuration;
6. train/freeze the final auxiliary head used for TTA tuning.

The Aux Head is **not retrained separately for every TTA candidate**.

This avoids mixing auxiliary-head tuning with TTA-method tuning.

---

# 16. TTA Hyperparameter Tuning

TTA hyperparameters are tuned using the:

```text
pseudo-OOD 10% validation streams
```

For every pseudo-OOD year, a frozen reference is evaluated.

For a TTA configuration:

\[
\Delta_t =
Acc_,t}
-------

Acc_{\mathrm{Frozen},t}
\]

The selection score is the mean improvement across pseudo-OOD years:

\[
\frac{1}{N}
\sum_t \Delta_t
\]

The configuration maximizing this score is selected.

---

# 17. Hyperparameter Inheritance

Some TTA variants inherit hyperparameters from simpler methods rather than being tuned independently.

The intended hierarchy is:

```text
metadata_cumulative
    │
    ├── metadata_cumulative_annual_reset
    │
    ├── metadata_cumulative_drift_reset
    │       │
    │       └── metadata_cumulative_drift_annual_reset
    │
    └── temporal_gradient_ema
```

In particular:

### Metadata Episodic

Tunes its own:

```text
learning_rate
normalized_aux_loss_threshold
```

### Metadata Cumulative

Tunes its own:

```text
learning_rate
normalized_aux_loss_threshold
```

### Cumulative + Annual Reset

Inherits the selected cumulative:

```text
learning_rate
threshold
```

No additional HP is required.

### Cumulative + Drift Reset

Inherits cumulative:

```text
learning_rate
threshold
```

and tunes:

```text
adwin_delta
```

### Cumulative + Drift + Annual Reset

Inherits both:

```text
cumulative learning rate / threshold
best drift configuration
```

### Temporal Gradient EMA

Inherits the selected cumulative metadata core:

```text
learning_rate
threshold
```

and tunes combinations of:

```text
beta
orthogonal_scale
warmup_updates
regularization
```

### TENT

Tunes:

```text
learning_rate
batch_size
```

---

# 18. Final Retraining

A `tune_and_test` experiment does **not** continue from the models created during tuning.

After hyperparameter selection:

```text
TUNING MODEL
     │
     └── discarded as final model
```

Then:

```text
selected hyperparameters
        │
        ▼
NEW MODEL FROM SCRATCH
        │
        ▼
train on final source
        │
        ▼
train auxiliary head
        │
        ▼
final evaluation
```

This separation is important for experimental correctness.

No tuning checkpoint should be reused as the final experimental model.

---

# 19. TTA Methods

## 19.1 Frozen

No test-time adaptation.

The source model is evaluated directly.

This provides the main baseline against which TTA improvements are computed.

---

# 20. Metadata Episodic

For every sample:

```text
reset adapter to SOURCE
        │
        ▼
auxiliary forward
        │
        ▼
metadata loss
        │
        ▼
adapter update
        │
        ▼
forward SAME sample again
        │
        ▼
main prediction
        │
        ▼
score prediction
```

Before processing the next sample, the adapter is reset again.

Therefore adaptation from sample \(t\) does not persist to sample \(t+1\).

---

# 21. Metadata Cumulative

Cumulative adaptation does not reset between samples.

For sample \(t\):

```text
adapter contains history from samples < t
                │
                ▼
        auxiliary update on t
                │
                ▼
       adapter after update t
                │
                ▼
       main prediction on t
```

Therefore the current prediction benefits from:

1. adaptation accumulated from previous samples;
2. adaptation from the current sample itself.

---

# 22. Metadata Cumulative + Annual Reset

This method behaves like cumulative adaptation within a year.

At a year boundary:

```text
previous year
     │
     ▼
YEAR BOUNDARY
     │
     ├── adapter  -> SOURCE
     └── optimizer -> FRESH
     │
     ▼
new year
```

The first year does not count as a reset event.

The reset occurs only when moving from one year to another.

---

# 23. Drift Detection

The drift-aware methods use ADWIN.

The current drift signal is:

```text
normalized_aux_loss
```

Support for signals such as main-prediction entropy can be added in the future.

## 23.1 Normalized auxiliary loss

The metadata cross-entropy loss is normalized as:

\[
L_}
===

\frac{L_{\mathrm{aux}}}
{\log(K_{\mathrm{aux}})}
\]

where \(K_{\mathrm{aux}}\) is the number of auxiliary classes.

This normalized loss is also used by the adaptation threshold.

## 23.2 Adaptation threshold

Adaptation occurs when:

\[
L_{\mathrm{norm}}
\ge
\text{normalized\_aux\_loss\_threshold}
\]

Therefore:

```text
threshold = 0.0
```

means that adaptation is effectively always enabled.

---

# 24. Metadata Cumulative + Drift Reset

The drift-triggering sample has precise semantics.

For sample \(t\):

```text
sample t
   │
   ▼
compute auxiliary loss BEFORE update
   │
   ▼
normalize auxiliary loss
   │
   ▼
send value to ADWIN
   │
   ├── NO DRIFT ───────────────────┐
   │                               │
   └── DRIFT                       │
       │                           │
       ▼                           │
 adapter -> SOURCE                 │
 optimizer -> FRESH                │
 ADWIN window -> EMPTY             │
       │                           │
       └───────────────────────────┤
                                   ▼
                         adapt using sample t
                                   │
                                   ▼
                         forward sample t again
                                   │
                                   ▼
                           main prediction
```

This means the sample that triggers drift is treated as the **first sample of the new regime**.

The algorithm does **not**:

```text
adapt t -> detect drift -> reset -> discard adaptation
```

Instead it does:

```text
detect drift -> reset -> adapt using t -> predict t
```

---

# 25. Metadata Cumulative + Drift + Annual Reset

This combines both reset mechanisms.

A drift reset performs:

```text
adapter       -> SOURCE
optimizer     -> FRESH
ADWIN window  -> EMPTY
```

At every year boundary it also performs:

```text
adapter       -> SOURCE
optimizer     -> FRESH
ADWIN window  -> EMPTY
```

ADWIN lifetime diagnostic counters can be preserved across annual window resets.

---

# 26. Temporal Gradient EMA

Temporal Gradient EMA uses metadata gradients while maintaining a temporal exponential moving average of previous metadata-gradient information.

The method includes:

```text
beta
orthogonal_scale
warmup_updates
regularization
gradient_clip
```

The EMA available for sample \(t\) represents historical information from earlier updates rather than future information.

The method inherits its metadata learning rate and normalized auxiliary-loss threshold from the selected `metadata_cumulative` configuration.

The fixed/default gradient clipping configuration is:

```text
gradient_clip = 0.1
steps = 1
```

Candidate tuning includes multiple regularization regimes rather than assuming a single legacy value.

---

# 27. TENT

TENT is fundamentally different from metadata-guided adaptation.

It minimizes entropy of the **main predictions**.

Only BatchNorm affine parameters are adapted:

```text
gamma
beta
```

The rest of the model remains frozen.

TENT uses batches rather than the samplewise metadata stream.

---

# 28. Evaluation Order

This distinction is critical.

There is deliberately **no universal prediction/update order for all TTA methods**.

The evaluator uses an explicit `EvaluationOrder` contract.

## 28.1 Metadata methods

Metadata methods use:

```text
ADAPT_THEN_PREDICT
```

For sample \(t\):

```text
x_t
 │
 ▼
Aux Head
 │
 ▼
auxiliary CE using aux_target_t
 │
 ▼
adapter update
 │
 ▼
forward x_t AGAIN
 │
 ▼
Main Head
 │
 ▼
main prediction
 │
 ▼
accuracy using main_target_t
```

The main-task prediction is therefore **post-adaptation for the current sample**.

## 28.2 TENT

TENT uses:

```text
PREDICT_THEN_ADAPT
```

For a batch:

```text
batch t
  │
  ▼
main prediction
  │
  ├──> prediction is scored
  │
  ▼
entropy
  │
  ▼
TENT update
  │
  ▼
affects future batches
```

The main prediction being evaluated is therefore **pre-update for the current batch**.

---

# 29. Supervised Reference

The supervised reference is **not a TTA method**.

It is an oracle/reference showing what happens when real OOD main labels are available for supervised adaptation.

For each OOD year independently:

```text
same pristine FINAL SOURCE MODEL
              │
              ▼
      deepcopy for year t
              │
              ▼
 train on OOD year t 70%
 using true MAIN labels
              │
              ▼
 early stopping on 10% validation
              │
              ▼
 evaluate on 20% test
```

Each OOD year is independent.

For example:

```text
2013 does NOT update the starting point of 2014.
```

Instead:

```text
final source ──> copy ──> supervised 2013
      │
      ├────────> copy ──> supervised 2014
      │
      ├────────> copy ──> supervised 2015
      │
      └────────> ...
```

Results are stored separately from TTA results.

---

# 30. Reproducibility

V3 distinguishes two seeds.

## 30.1 `experiment_seed`

Controls stochastic behavior associated with model training and experimental execution.

Examples include model initialization and other seeded stochastic training operations.

## 30.2 `split_seed`

Controls dataset split membership.

It determines which samples belong to:

```text
70% train
10% validation
20% test
```

The same `split_seed` should reproduce the same split.

## 30.3 Why keep them separate?

This allows experiments such as:

```text
experiment_seed=42
split_seed=42
```

or:

```text
experiment_seed=43
split_seed=42
```

where the data partition is unchanged but training stochasticity changes.

It also permits both to vary.

---

# 31. Experiment Modes

Three modes are supported.

## 31.1 `tune`

Runs hyperparameter tuning only.

```bash
python scripts/run.py \
  --dataset-config configs/datasets/fmow.yaml \
  --experiment-config configs/experiments/default.yaml \
  --tuning-config configs/tuning/default.yaml \
  --mode tune \
  --experiment-seed 42 \
  --split-seed 42
```

## 31.2 `test`

Runs the final experiment using explicitly supplied or previously selected hyperparameters.

Example interface:

```bash
python scripts/run.py \
  --dataset-config configs/datasets/fmow.yaml \
  --experiment-config configs/experiments/default.yaml \
  --mode test \
  --experiment-seed 42 \
  --split-seed 42 \
  --best-overrides PATH_TO_BEST_OVERRIDES
```

## 31.3 `tune_and_test`

This is the intended mode for complete final experiments.

It performs:

```text
tune
 │
 ▼
select HPs
 │
 ▼
discard tuning models
 │
 ▼
restart from scratch
 │
 ▼
final source training
 │
 ▼
final OOD evaluation
```

Example:

```bash
python scripts/run.py \
  --dataset-config configs/datasets/fmow.yaml \
  --experiment-config configs/experiments/default.yaml \
  --tuning-config configs/tuning/default.yaml \
  --mode tune_and_test \
  --experiment-seed 42 \
  --split-seed 42
```

---

# 32. Running a Single Experiment

From the repository root:

```bash
conda activate metadata_tta
```

Then:

```bash
python scripts/run.py \
  --dataset-config configs/datasets/fmow.yaml \
  --experiment-config configs/experiments/default.yaml \
  --tuning-config configs/tuning/default.yaml \
  --mode tune_and_test \
  --experiment-seed 42 \
  --split-seed 42
```

The CLI can be inspected using:

```bash
python scripts/run.py --help
```

---

# 33. Batch Experiments

Multiple independent runs can be launched using:

```text
scripts/run_batch.py
```

Batch configuration files are stored in:

```text
configs/batches/
```

## 33.1 Seed matrix

The batch runner uses a two-column seed matrix.

Each row is:

```text
[experiment_seed, split_seed]
```

Example:

```yaml
dataset: fmow

mode: tune_and_test

seeds:
  - [42, 42]
  - [43, 43]
  - [44, 44]
  - [45, 45]
  - [46, 46]

dataset_config: configs/datasets/fmow.yaml
experiment_config: configs/experiments/default.yaml
tuning_config: configs/tuning/default.yaml

continue_on_error: false
```

This produces exactly five experiments:

```text
run 1 -> experiment_seed=42, split_seed=42
run 2 -> experiment_seed=43, split_seed=43
run 3 -> experiment_seed=44, split_seed=44
run 4 -> experiment_seed=45, split_seed=45
run 5 -> experiment_seed=46, split_seed=46
```

It does **not** produce a Cartesian product.

## 33.2 Different experiment and split seeds

This is also valid:

```yaml
seeds:
  - [42, 7]
  - [43, 7]
  - [44, 7]
```

Here all experiments use the same split but different experiment seeds.

## 33.3 Running a batch

```bash
python scripts/run_batch.py \
  --config configs/batches/fmow_5seeds.yaml
```

The runs are executed serially and independently.

If:

```yaml
continue_on_error: false
```

the batch stops on the first failed experiment.

---

# 34. Per-Seed Hyperparameter Tuning

Hyperparameters are **not tuned once on one seed and then reused across every run**.

For `tune_and_test`, every seed pair performs its own tuning.

For example:

```text
(42, 42)
   │
   ├── model tuning
   ├── aux tuning
   ├── TTA tuning
   └── final run


(43, 43)
   │
   ├── NEW model tuning
   ├── NEW aux tuning
   ├── NEW TTA tuning
   └── final run
```

The candidate configuration lists remain the same, but the winning candidate may differ across runs.

---

# 35. Output Structure

Results are organized by dataset and run.

Conceptually:

```text
results/
└── <dataset>/
    └── <experiment_seed>-<split_seed>-<timestamp>/
        │
        ├── run_config.yaml
        ├── manifest.json
        ├── run.log
        │
        ├── tuning/
        │   ├── model/
        │   │   ├── trials.csv
        │   │   └── best.yaml
        │   │
        │   ├── aux_head/
        │   │   ├── trials.csv
        │   │   └── best.yaml
        │   │
        │   ├── metadata_episodic/
        │   ├── metadata_cumulative/
        │   ├── metadata_cumulative_annual_reset/
        │   ├── metadata_cumulative_drift_reset/
        │   ├── metadata_cumulative_drift_annual_reset/
        │   ├── temporal_gradient_ema/
        │   ├── tent/
        │   └── cotta/
        │
        ├── checkpoints/
        │   ├── source_model.pt
        │   └── aux_head.pt
        │
        └── results/
            ├── id/
            │   └── frozen/
            │       ├── yearly_results.csv
            │       └── summary.json
            │
            ├── tta/
            │   ├── frozen/
            │   ├── metadata_episodic/
            │   ├── metadata_cumulative/
            │   ├── metadata_cumulative_annual_reset/
            │   ├── metadata_cumulative_drift_reset/
            │   ├── metadata_cumulative_drift_annual_reset/
            │   ├── temporal_gradient_ema/
            │   ├── tent/
            │   └── cotta/
            │
            ├── supervised_reference/
            │   ├── yearly_results.csv
            │   └── summary.json
            │
            └── summary.csv
```

---

# 36. Manifest

Every run stores a:

```text
manifest.json
```

containing run-level metadata such as:

```text
dataset
timestamp
git commit
experiment seed
split seed
mode
enabled methods
configuration paths
```

This makes it possible to trace an output directory back to the exact experimental setup.

---

# 37. Logging

Logging configuration is controlled through:

```yaml
logging:
  console: true
  save_to_file: true
  level: INFO
```

Logs are written to:

```text
run.log
```

Important information includes:

- dataset;
- mode;
- seeds;
- device;
- run directory;
- tuning phase;
- candidate ID;
- candidate parameters;
- validation results;
- selected configuration;
- final per-year evaluation;
- important reset/drift events and diagnostics.

The intention is to provide enough information to understand the execution without logging every individual sample during normal operation.

---

# 38. Correctness Checks

The V3 design includes explicit sanity checks for experimental correctness.

Important invariants include:

```text
train / validation / test are disjoint
```

```text
pseudo-OOD TTA tuning uses validation only
```

```text
test data are not used for hyperparameter tuning
```

```text
final model is retrained from scratch
```

```text
tuning checkpoints are not reused as final models
```

```text
Single Head and Double Head main predictions match
before Aux Head training
```

```text
only the intended parameters are trainable
```

```text
episodic adaptation resets for every sample
```

```text
cumulative adaptation persists across samples
```

```text
annual resets occur only at year boundaries
```

```text
ADWIN does not access main-task labels
```

```text
metadata TTA evaluates the current sample
AFTER metadata adaptation
```

```text
TENT evaluates the current batch BEFORE its entropy update
```

Violations of important invariants should raise errors rather than silently changing the protocol.

---

# 39. Smoke Test

Before launching expensive experiments, V3 can be tested using reduced configurations:

```text
configs/experiments/smoke.yaml
configs/tuning/smoke.yaml
```

The smoke experiment uses:

```text
1 training epoch
1 explicit candidate per tuned stage
```

while still traversing the real experimental pipeline.

Run:

```bash
python scripts/run.py \
  --dataset-config configs/datasets/fmow.yaml \
  --experiment-config configs/experiments/smoke.yaml \
  --tuning-config configs/tuning/smoke.yaml
```

The V3 pipeline has successfully completed this end-to-end smoke test on FMoW.

The smoke test is intended to verify software correctness and pipeline connectivity.

Its numerical results are **not scientifically meaningful** and should not be reported as experimental results.

---

# 40. Tuning Configuration

The full tuning search is defined in:

```text
configs/tuning/default.yaml
```

The search uses explicit candidate lists rather than automatically generating every possible Cartesian combination.

This keeps the computational budget controlled and makes every evaluated configuration explicit and auditable.

Approximate candidate counts in the current design are:

```text
model                                  15
aux_head                                4
metadata_episodic                      12
metadata_cumulative                    12
metadata_cumulative_annual_reset        1 inherited
metadata_cumulative_drift_reset         6
metadata_cumulative_drift_annual_reset  1 inherited
temporal_gradient_ema                  12
tent                                   12
cotta                                   0
```

CoTTA currently has no candidates because it is not implemented.

---

# 41. CoTTA

CoTTA is reserved in the V3 configuration and output structure but is currently disabled:

```yaml
cotta:
  enabled: false
```

It must not be enabled until its implementation and evaluation semantics are added.

The placeholder exists so that CoTTA can later be integrated without restructuring the entire experimental pipeline.

---

# 42. Adding a New Dataset

To add a dataset:

1. prepare the fixed embeddings;
2. provide `target`;
3. provide `aux_target`;
4. verify the year information required by the loader;
5. create:

```text
configs/datasets/<dataset>.yaml
```

6. specify:

```text
n_classes_main
n_classes_aux
pseudo_source
pseudo_ood
final_source
real_ood
```

7. verify the 70/10/20 split;
8. run protocol sanity checks;
9. run the smoke pipeline before launching full experiments.

Do not introduce a dataset directly into full multi-seed experiments before checking its split and temporal protocol.

---

# 43. Adding a New TTA Method

A new TTA method should implement the common TTA interface.

Most importantly, it must explicitly declare its evaluation order.

Possible semantics include:

```text
ADAPT_THEN_PREDICT
```

or:

```text
PREDICT_THEN_ADAPT
```

The method should also expose its required stream batch size.

After implementation:

1. add the method to the TTA registry;
2. add its configuration under `methods`;
3. add tuning candidates if required;
4. add result output support;
5. add protocol/evaluation sanity checks;
6. verify trainable parameters;
7. verify evaluation order;
8. run the smoke test.

---

# 44. Experimental Workflow

For a complete experiment, the high-level workflow is:

```text
                     DATASET
                        │
                        ▼
                 temporal splitting
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
    pseudo-source                 pseudo-OOD
          │                           │
          ▼                           │
    MODEL TUNING                      │
          │                           │
          ▼                           │
     best model HP                    │
          │                           │
          ▼                           │
 train pseudo-source model            │
          │                           │
          ▼                           │
    AUX HEAD TUNING                   │
          │                           │
          ▼                           │
     best aux HP                      │
          │                           │
          ▼                           │
 train/freeze Aux Head                │
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
                    TTA TUNING
                pseudo-OOD val only
                        │
                        ▼
                selected TTA HPs
                        │
                        ▼
              DISCARD TUNING MODELS
                        │
                        ▼
                  START FROM SCRATCH
                        │
                        ▼
              train FINAL SOURCE
                        │
                        ▼
                 train Aux Head
                        │
                        ▼
             REAL-OOD TEST STREAM
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
     Frozen        TTA methods      Supervised
                                    Reference
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
                 final results
```

---

# 45. Scientific Interpretation

The methods answer different questions.

### Frozen

> How well does the source model generalize without adaptation?

### Metadata TTA

> Can metadata available at test time improve the representation without using main-task labels?

### Episodic

> Is adapting independently to each current sample useful?

### Cumulative

> Is accumulating metadata adaptation over the temporal stream useful?

### Annual Reset

> Does preventing adaptation from propagating indefinitely across calendar years help?

### Drift Reset

> Can an online drift detector identify when accumulated adaptation should be discarded?

### Temporal Gradient EMA

> Can historical metadata-gradient information provide a better adaptation direction?

### TENT

> How does metadata-guided adaptation compare with unsupervised entropy-minimization TTA?

### Supervised Reference

> What performance is achievable when OOD main labels are actually available for supervised adaptation?

The supervised reference must therefore remain clearly separated from actual TTA methods.

---

# 46. Important Experimental Rules

The following rules should be preserved when modifying the repository.

**Rule 1 — Never tune on real test data.**

```text
TTA tuning = pseudo-OOD validation only
```

**Rule 2 — Main labels must never drive metadata TTA updates.**

They are used only to compute evaluation metrics.

**Rule 3 — Final training starts from scratch after tuning.**

Do not promote a tuning checkpoint into the final experiment.

**Rule 4 — Metadata methods predict after adapting to the current sample.**

```text
aux -> update -> same sample again -> main prediction
```

**Rule 5 — TENT predicts before adapting on the current batch.**

```text
main prediction -> score -> entropy update
```

**Rule 6 — Drift is detected before adapting the drift-triggering sample.**

```text
detect -> reset -> adapt current sample -> predict
```

**Rule 7 — Annual reset means reset at a year boundary.**

It is not a generic evaluator reset flag.

**Rule 8 — Every reset returns the metadata adapter to the source state.**

This keeps reset ablations interpretable.

**Rule 9 — Experiment and split seeds are independent concepts.**

Always record both.

**Rule 10 — Every final run performs its own tuning when using `tune_and_test`.**

Do not select hyperparameters on one seed and silently reuse them across all other final runs.

---

# 47. Current Status

The V3 pipeline currently supports:

```text
[implemented] Frozen
[implemented] Supervised Reference
[implemented] Metadata Episodic
[implemented] Metadata Cumulative
[implemented] Metadata Cumulative + Annual Reset
[implemented] Metadata Cumulative + Drift Reset
[implemented] Metadata Cumulative + Drift + Annual Reset
[implemented] Temporal Gradient EMA
[implemented] TENT
[placeholder] CoTTA
```

The V3 pipeline has successfully completed an end-to-end smoke run on FMoW.

Before large experimental campaigns, the recommended sequence is:

```text
1. static checks
2. smoke test
3. inspect output files
4. inspect logs
5. run one complete full experiment
6. inspect results
7. launch multi-seed batch experiments
```

---

# 48. Quick Start

For a new machine:

```bash
git clone <repository-url>
cd metadata_tta_v3
```

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate metadata_tta
```

Check the CLI:

```bash
python scripts/run.py --help
```

Run a smoke experiment:

```bash
python scripts/run.py \
  --dataset-config configs/datasets/fmow.yaml \
  --experiment-config configs/experiments/smoke.yaml \
  --tuning-config configs/tuning/smoke.yaml
```

Run one full experiment:

```bash
python scripts/run.py \
  --dataset-config configs/datasets/fmow.yaml \
  --experiment-config configs/experiments/default.yaml \
  --tuning-config configs/tuning/default.yaml \
  --mode tune_and_test \
  --experiment-seed 42 \
  --split-seed 42
```

Run a batch:

```bash
python scripts/run_batch.py \
  --config configs/batches/fmow_5seeds.yaml
```

Inspect:

```text
results/<dataset>/<experiment_seed>-<split_seed>-<timestamp>/
```

for the configuration, manifest, logs, tuning results, checkpoints, and final evaluation results.
