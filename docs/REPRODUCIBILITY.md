# VLCEA reproducibility checklist

Before running the full experiments, verify the following.

## 1. Data layout

```text
data/j30/j30.zip
data/j60/j60.zip
data/j90/j90.zip
data/j120/j120.zip

data/j30/bks_j30.csv
data/j60/bks_j60.csv
data/j90/bks_j90.csv
data/j120/bks_j120.csv
```

## 2. LLM setup

```bash
ollama pull qwen2.5:7b-instruct
ollama ps
```

The reported protocol uses:

```text
model = qwen2.5:7b-instruct
temperature = 0
num_ctx = 8192
num_predict = 4096
max_attempts = 6
```

## 3. Constraint generation and validation

```bash
python code/constraint_synthesis/generate_local_constraints_ollama_hybrid_edges_v2.py --dataset j120 --fallback_det_scores --overwrite
python code/constraint_synthesis/validate_constraints.py results/constraints/j120/out_local_qwen25
```

Expected full instance counts:

```text
J30: 480 JSON files
J60: 480 JSON files
J90: 480 JSON files
J120: 600 JSON files
```

## 4. EA settings

Reported fixed-budget setting:

```text
population P = 10
generations G = 8
NFE = P(G+1) = 90
seeds = 0-29
beta = 0.5
gamma = 0.2
lambda1 = 0.05
lambda2 = 0.01
hard_conf = 0.9
```

## 5. Required output files for paper tables

```text
results/ea_runs/<dataset>/results_seed0.csv ... results_seed29.csv
results/ea_runs/<dataset>/summary_multiseed.csv
results/ea_runs/<dataset>/win_tie_loss_vs_HC.csv
results/ea_runs/<dataset>/wilcoxon_vs_HC.csv
results/runtime/<dataset>_runtime_summary.csv
```

Note: For J60, J90, and J120, the files named `bks_j60.csv`, `bks_j90.csv`, and `bks_j120.csv` are code-compatible reference files that store PSPLIB heuristic reference solution (HRS) makespans. They should be used for `RPD to HRS` and `HRS-or-better rate`, not interpreted as proven optimal or best-known values.