#!/usr/bin/env python3
"""
Aggregate VLCEA multi-seed RCPSP results into paper-ready CSV files.

Supports the hierarchical project layout:
  data/<dataset>/bks_<dataset>.csv
  results/ea_runs/<dataset>/results_seed*.csv
  results/constraints/<dataset>/out_local_qwen25/failures.txt

Input result CSVs must contain:
  instance, seed, bks, NC, HC, LC_A, LC_B, LC_C

For J30, the bks column is treated as BKS/OPT.
For J60/J90/J120, if HRS values are stored in the bks column for code
compatibility, run with --reference_label HRS --hit_mode le.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

METHODS = ["NC", "HC", "LC_A", "LC_B", "LC_C"]


def _sort_seed_files(files: Iterable[str]) -> list[str]:
    def key(path: str) -> int:
        stem = Path(path).stem
        try:
            return int(stem.replace("results_seed", ""))
        except ValueError:
            return 10**9
    return sorted(files, key=key)


def infer_paths(args):
    root = Path(args.project_root).resolve()
    ds = args.dataset.lower() if args.dataset else None

    results_dir = Path(args.results_dir) if args.results_dir else None
    if results_dir is None:
        if not ds:
            raise ValueError("Either --dataset or --results_dir must be provided")
        results_dir = root / "results" / "ea_runs" / ds
    elif not results_dir.is_absolute():
        results_dir = root / results_dir

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is None:
        out_dir = results_dir
    elif not out_dir.is_absolute():
        out_dir = root / out_dir

    failures_txt = Path(args.failures_txt) if args.failures_txt else None
    if failures_txt is None and ds:
        cand = root / "results" / "constraints" / ds / args.constraints_name / "failures.txt"
        if cand.exists():
            failures_txt = cand
    elif failures_txt is not None and not failures_txt.is_absolute():
        failures_txt = root / failures_txt

    ref_label = args.reference_label
    if ref_label is None:
        ref_label = "BKS" if ds == "j30" else "HRS"
    ref_label = ref_label.upper()

    hit_mode = args.hit_mode
    if hit_mode == "auto":
        hit_mode = "equal" if ref_label == "BKS" else "le"

    return results_dir, out_dir, failures_txt, ref_label, hit_mode


def _load_results(results_dir: Path) -> pd.DataFrame:
    files = _sort_seed_files(glob.glob(str(results_dir / "results_seed*.csv")))
    if not files:
        raise FileNotFoundError(f"No results_seed*.csv files found in: {results_dir}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if "LC_B" not in df.columns and "LC" in df.columns:
            df["LC_B"] = df["LC"]

        required = ["instance", "seed", "bks"] + METHODS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} is missing required columns: {missing}")

        # Recompute RPD to the reference stored in column bks.
        for m in METHODS:
            df[f"rpd_{m}_pct"] = (df[m] - df["bks"]) / df["bks"] * 100.0
            # Backward-compatible gap column name.
            df[f"gap_{m}_pct"] = df[f"rpd_{m}_pct"]

        keep = ["instance", "seed", "bks"] + METHODS + [f"rpd_{m}_pct" for m in METHODS] + [f"gap_{m}_pct" for m in METHODS]
        dfs.append(df[keep])

    return pd.concat(dfs, ignore_index=True)


def _hit_series(g: pd.DataFrame, method: str, hit_mode: str) -> pd.Series:
    if hit_mode == "equal":
        return g[method] == g["bks"]
    if hit_mode == "le":
        return g[method] <= g["bks"]
    raise ValueError("hit_mode must be equal/le/auto")


def write_per_seed_summary(all_df: pd.DataFrame, out_dir: Path, ref_label: str, hit_mode: str) -> pd.DataFrame:
    hit_name = f"{ref_label}_hit_rate_pct" if hit_mode == "equal" else f"{ref_label}_or_better_rate_pct"
    rows = []
    for seed, g in all_df.groupby("seed", sort=True):
        row = {"seed": int(seed), "n_instances": int(g["instance"].nunique()), "reference_label": ref_label, "hit_mode": hit_mode}
        for m in METHODS:
            row[f"Mean_makespan_{m}"] = float(g[m].mean())
            row[f"ARPD_{m}_pct"] = float(g[f"rpd_{m}_pct"].mean())
            row[f"{hit_name}_{m}"] = float(_hit_series(g, m, hit_mode).mean() * 100.0)
            # Generic alias for easier plotting.
            row[f"Hit_rate_{m}_pct"] = row[f"{hit_name}_{m}"]
            row[f"Median_RPD_{m}_pct"] = float(g[f"rpd_{m}_pct"].median())
        rows.append(row)
    per_seed = pd.DataFrame(rows).sort_values("seed")
    per_seed.to_csv(out_dir / "per_seed_summary.csv", index=False)
    return per_seed


def write_per_instance_mean(all_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    agg_spec = {"bks": ("bks", "first")}
    for m in METHODS:
        agg_spec[m] = (m, "mean")
    per_inst = all_df.groupby("instance").agg(**agg_spec).reset_index()
    for m in METHODS:
        per_inst[f"rpd_{m}_pct"] = (per_inst[m] - per_inst["bks"]) / per_inst["bks"] * 100.0
        per_inst[f"gap_{m}_pct"] = per_inst[f"rpd_{m}_pct"]
    per_inst.to_csv(out_dir / "per_instance_mean.csv", index=False)
    return per_inst


def write_summary_multiseed(per_seed: pd.DataFrame, per_inst: pd.DataFrame, out_dir: Path, ref_label: str, hit_mode: str) -> pd.DataFrame:
    hit_name = f"{ref_label}_hit_rate_pct" if hit_mode == "equal" else f"{ref_label}_or_better_rate_pct"
    rows = []
    for m in METHODS:
        rows.append({
            "method": m,
            "reference_label": ref_label,
            "hit_mode": hit_mode,
            "Mean_makespan_mean": per_seed[f"Mean_makespan_{m}"].mean(),
            "Mean_makespan_std": per_seed[f"Mean_makespan_{m}"].std(ddof=1),
            f"ARPD_to_{ref_label}_pct_mean": per_seed[f"ARPD_{m}_pct"].mean(),
            f"ARPD_to_{ref_label}_pct_std": per_seed[f"ARPD_{m}_pct"].std(ddof=1),
            f"{hit_name}_mean": per_seed[f"Hit_rate_{m}_pct"].mean(),
            f"{hit_name}_std": per_seed[f"Hit_rate_{m}_pct"].std(ddof=1),
            f"Median_RPD_to_{ref_label}_pct": per_inst[f"rpd_{m}_pct"].median(),
            # Generic aliases.
            "ARPD_to_ref_pct_mean": per_seed[f"ARPD_{m}_pct"].mean(),
            "ARPD_to_ref_pct_std": per_seed[f"ARPD_{m}_pct"].std(ddof=1),
            "Hit_rate_pct_mean": per_seed[f"Hit_rate_{m}_pct"].mean(),
            "Hit_rate_pct_std": per_seed[f"Hit_rate_{m}_pct"].std(ddof=1),
            "Median_RPD_to_ref_pct": per_inst[f"rpd_{m}_pct"].median(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary_multiseed.csv", index=False)
    return summary


def write_wtl_and_wilcoxon(per_inst: pd.DataFrame, out_dir: Path) -> None:
    wtl_rows = []
    wil_rows = []
    hc_makespan = per_inst["HC"].to_numpy()
    hc_rpd = per_inst["rpd_HC_pct"].to_numpy()

    for m in ["NC", "LC_A", "LC_B", "LC_C"]:
        vals = per_inst[m].to_numpy()
        wtl_rows.append({
            "method_vs_HC": m,
            "win": int(np.sum(vals < hc_makespan)),
            "tie": int(np.sum(vals == hc_makespan)),
            "loss": int(np.sum(vals > hc_makespan)),
        })
        if wilcoxon is None:
            stat, p = np.nan, np.nan
        else:
            x_rpd = per_inst[f"rpd_{m}_pct"].to_numpy()
            try:
                stat, p = wilcoxon(x_rpd, hc_rpd, alternative="two-sided", zero_method="wilcox")
            except Exception:
                stat, p = np.nan, np.nan
        wil_rows.append({"method_vs_HC": m, "wilcoxon_stat": stat, "p_value": p})

    pd.DataFrame(wtl_rows).to_csv(out_dir / "win_tie_loss_vs_HC.csv", index=False)
    pd.DataFrame(wil_rows).to_csv(out_dir / "wilcoxon_vs_HC.csv", index=False)


def write_fallback_files(failures_txt: Path | None, out_dir: Path) -> None:
    if failures_txt is None:
        return
    if not failures_txt.exists():
        print(f"[WARN] failures_txt not found: {failures_txt}")
        return
    lines = [line.strip() for line in failures_txt.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    pd.DataFrame({"fallback_instances": lines}).to_csv(out_dir / "fallback_instances.csv", index=False)
    pd.DataFrame([{"fallback_count": len(lines)}]).to_csv(out_dir / "fallback_meta.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".", help="Project root containing data/ and results/")
    parser.add_argument("--dataset", default=None, choices=["j30", "j60", "j90", "j120"], help="Dataset key for hierarchical layout")
    parser.add_argument("--results_dir", default=None, help="Directory containing results_seed*.csv. Default: results/ea_runs/<dataset>")
    parser.add_argument("--out_dir", default=None, help="Output directory. Default: same as results_dir")
    parser.add_argument("--constraints_name", default="out_local_qwen25", help="For auto-detecting failures.txt")
    parser.add_argument("--failures_txt", default=None, help="Optional failures.txt from constraint generation")
    parser.add_argument("--reference_label", default=None, help="BKS for J30, HRS for J60/J90/J120")
    parser.add_argument("--hit_mode", default="auto", choices=["auto", "equal", "le"], help="equal: Cmax==ref, le: Cmax<=ref")
    args = parser.parse_args()

    results_dir, out_dir, failures_txt, ref_label, hit_mode = infer_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] results_dir={results_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] reference_label={ref_label}, hit_mode={hit_mode}")
    if failures_txt:
        print(f"[INFO] failures_txt={failures_txt}")

    all_df = _load_results(results_dir)
    per_seed = write_per_seed_summary(all_df, out_dir, ref_label, hit_mode)
    per_inst = write_per_instance_mean(all_df, out_dir)
    write_summary_multiseed(per_seed, per_inst, out_dir, ref_label, hit_mode)
    write_wtl_and_wilcoxon(per_inst, out_dir)
    write_fallback_files(failures_txt, out_dir)

    print(f"[OK] Aggregated {all_df['seed'].nunique()} seeds and {all_df['instance'].nunique()} instances.")
    print(f"[OK] Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
