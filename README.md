# VLCEA

**VLCEA: LLM-Guided Evolutionary Scheduling for RCPSP via Validated, Hallucination-Safe Constraints**

This repository contains the public implementation of VLCEA, a framework that uses a Large Language Model (LLM) to synthesize validated, instance-grounded guidance for evolutionary scheduling on the Resource-Constrained Project Scheduling Problem (RCPSP).

VLCEA uses the LLM only to generate activity-level priority scores. Ordering hints are then constructed deterministically from direct precedence relations already present in the instance, preventing hallucinated precedence constraints by design. The validated guidance is injected into a permutation-based evolutionary algorithm with an SSGS decoder through repair-level, decoder-level, and fitness-level mechanisms.

## Repository structure

```text
VLCEA/
  code/
    constraint_synthesis/
      generate_local_constraints_ollama_hybrid_edges_v2.py
      generate_strict_v3_constraints.py
      validate_constraints.py
      prompt_template_scores_only_v3_min.txt
      rcpsp_mixed_constraint_schema.json
    ea/
      run_rcpsp_experiment.py
    analysis/
      aggregate_multiseed_results.py
      summarize_runtime.py
      aggregate_rcpsp_results.py
      make_paper_tables.py
  data/
    j30/
    j60/
    j90/
    j120/
  results/
  docs/
  requirements.txt
```

The `results/` directory is intentionally ignored by Git. Generated constraints, EA outputs, runtime logs, and raw result CSVs should be stored there during local runs, but are not included in the code repository.

## Environment

Recommended:

- Python 3.10+
- Ollama for LLM-based constraint generation
- `qwen2.5:7b-instruct` for reproducing the reported VLCEA protocol

Install Python dependencies:

```bash
conda create -n vlcea python=3.10 -y
conda activate vlcea
pip install -r requirements.txt
```

Install and test the Ollama model:

```bash
ollama pull qwen2.5:7b-instruct
ollama run qwen2.5:7b-instruct
```

The code calls the Ollama API at `http://localhost:11434` by default.

## Data preparation

Download PSPLIB RCPSP instances externally and place the zip files as follows:

```text
data/j30/j30.zip
data/j60/j60.zip
data/j90/j90.zip
data/j120/j120.zip
```

Reference makespan CSV files should be placed as:

```text
data/j30/bks_j30.csv
data/j60/bks_j60.csv
data/j90/bks_j90.csv
data/j120/bks_j120.csv
```

Each reference CSV must use the following format:

```csv
instance,bks
j3010_1.sm,43
j3010_2.sm,47
...
```

For J30, the `bks` column stores the BKS/OPT reference. For J60, J90, and J120, this repository keeps the same column name for code compatibility, but the values store PSPLIB heuristic reference solution (HRS) makespans for code compatibility rather than proven optima. In publications, report these larger-set metrics as `RPD to HRS` and `HRS-or-better rate`, not as optimality gaps.

## Step 1: Generate validated constraints

Example for J120:

```bash
python code/constraint_synthesis/generate_local_constraints_ollama_hybrid_edges_v2.py \
  --dataset j120 \
  --constraints_name out_local_qwen25 \
  --model qwen2.5:7b-instruct \
  --llm_id ollama/qwen2.5:7b-instruct \
  --temperature 0 \
  --seed 0 \
  --num_ctx 8192 \
  --num_predict 4096 \
  --timeout_sec 1200 \
  --max_attempts 6 \
  --fallback_det_scores \
  --overwrite
```

Output:

```text
results/constraints/j120/out_local_qwen25/*.json
results/constraints/j120/out_local_qwen25/generation_time_log.csv
results/constraints/j120/out_local_qwen25/failures.txt
results/constraints/j120/out_local_qwen25_raw/*.attempt*.txt
```

The final JSON files in `out_local_qwen25/` are the files used by the EA. The `out_local_qwen25_raw/` directory contains raw LLM attempts for auditing.

## Step 2: Validate constraints

```bash
python code/constraint_synthesis/validate_constraints.py results/constraints/j120/out_local_qwen25
```

Expected output for the full J120 set:

```text
OK: 600 files validated
```

## Step 3: Run the EA experiments

```bash
python code/ea/run_rcpsp_experiment.py \
  --dataset j120 \
  --constraints_name out_local_qwen25 \
  --seeds 0-29 \
  --pop 10 \
  --gen 8 \
  --beta 0.5 \
  --gamma 0.2 \
  --lambda1 0.05 \
  --lambda2 0.01 \
  --hard_conf 0.9
```

Output:

```text
results/ea_runs/j120/results_seed0.csv
...
results/ea_runs/j120/results_seed29.csv
results/ea_runs/j120/ea_runtime_log.csv
results/ea_runs/j120/ea_runtime_summary.csv
```

Compared methods:

- `NC`: no additional guidance
- `HC`: hand-crafted/topological repair baseline
- `LC_A`: repair-level injection
- `LC_B`: decoder-bias injection (`LC` alias)
- `LC_C`: fitness-penalty injection

## Step 4: Aggregate results

For J30:

```bash
python code/analysis/aggregate_multiseed_results.py \
  --dataset j30 \
  --reference_label BKS \
  --hit_mode equal
```

For J60/J90/J120 with HRS references:

```bash
python code/analysis/aggregate_multiseed_results.py \
  --dataset j120 \
  --reference_label HRS \
  --hit_mode le
```

Output files are written to `results/ea_runs/<dataset>/`:

```text
summary_multiseed.csv
per_seed_summary.csv
per_instance_mean.csv
win_tie_loss_vs_HC.csv
wilcoxon_vs_HC.csv
fallback_instances.csv
fallback_meta.csv
```

## Step 5: Summarize runtime logs

```bash
python code/analysis/summarize_runtime.py \
  --dataset j120 \
  --constraints_name out_local_qwen25 \
  --seeds 30
```

Output:

```text
results/runtime/j120_runtime_summary.csv
```

This summarizes constraint-generation time and EA method-level runtime from the generated logs.

## Notes on reproducibility

- The LLM-generated constraints are stochastic only to the extent permitted by the local inference stack. The reported protocol uses `temperature=0`, fixed generation settings, schema validation, bounded retries, and deterministic fallback.
- For exact reproduction of published numerical results, use the archived generated constraints and raw result CSVs supplied separately through the paper's data repository or supplementary data.
- Re-running LLM constraint generation may produce small differences across Ollama/model versions or hardware environments, even with deterministic settings.
- J60/J90/J120 reference values should be interpreted as HRS references unless proven BKS/optimal values are used.

## Citation

If you use this code, please cite the accompanying paper:

```text
H. Kim, W. Song, and H. Kim, "VLCEA: LLM-Guided Evolutionary Scheduling for RCPSP via Validated, Hallucination-Safe Constraints."
```

## License

The source code in this repository is made available for non-commercial research and educational use under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted without prior written permission from the authors.

Generated experimental artifacts, including mixed-constraint JSON files, aggregated result tables, runtime logs, and analysis outputs, are released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license unless otherwise stated.

The original PSPLIB benchmark instances are not redistributed in this repository. Users must obtain PSPLIB instances from the official PSPLIB source and follow its terms of use.

This repository is source-available for non-commercial research use and should not be described as OSI-approved open source.