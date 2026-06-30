#!/usr/bin/env python3
"""
Aggregate per-seed RCPSP results into paper-ready summaries + LaTeX tables.

Expected input files: results_seed*.csv with fixed columns:
instance,param,inst,bks,NC,HC,LC,LC_A,LC_B,LC_C,seed

This script will:
- validate columns and recompute gaps
- create per-seed summary_seed{seed}.csv
- create multiseed summary_multiseed.csv (mean/std over seeds)
- create LaTeX table_main.tex

Usage:
  python aggregate_rcpsp_results.py --input_glob "results_seed*.csv" --out_dir out
"""
import argparse, glob, os
import numpy as np
import pandas as pd

METHODS = ["NC","HC","LC","LC_A","LC_B","LC_C"]

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bks = df["bks"].to_numpy()
    for m in METHODS:
        vals = df[m].to_numpy()
        gaps = (vals - bks) / bks * 100.0
        opt_hits = int(np.sum(vals == bks))
        unsolved = gaps[gaps > 0]
        rows.append({
            "method": m,
            "mean_Cmax": float(np.mean(vals)),
            "std_Cmax": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "ARPD_pct": float(np.mean(gaps)),
            "median_RPD_pct": float(np.median(gaps)),
            "opt_hits": opt_hits,
            "opt_hit_rate_pct": float(opt_hits / len(df) * 100.0),
            "mean_RPD_unsolved_pct": float(np.mean(unsolved)) if len(unsolved) else 0.0,
            "worst_RPD_pct": float(np.max(gaps)),
        })
    return pd.DataFrame(rows)

def make_latex_table(mult: pd.DataFrame) -> str:
    # expects columns: method, ARPD_mean, ARPD_std, opt_hit_rate_mean, etc.
    # For simplicity we build from multiseed summary produced below.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RCPSP results on PSPLIB J30 (480 instances). Reported values are mean$\pm$std over seeds. RPD is relative percentage deviation to the best-known (optimal for J30) makespan.}")
    lines.append(r"\label{tab:j30_multiseed}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Mean $C_{\max}$ & ARPD (\%) & Opt.\ hit rate (\%) & Worst RPD (\%)\\")
    lines.append(r"\midrule")
    for _, r in mult.iterrows():
        m = r["method"]
        mean_c = r["mean_Cmax_mean"]
        std_c = r["mean_Cmax_std"]
        arpd_m = r["ARPD_pct_mean"]
        arpd_s = r["ARPD_pct_std"]
        hit_m = r["opt_hit_rate_pct_mean"]
        hit_s = r["opt_hit_rate_pct_std"]
        worst_m = r["worst_RPD_pct_mean"]
        worst_s = r["worst_RPD_pct_std"]
        def pm(a,b):
            if np.isnan(b):
                return f"{a:.3f}"
            return f"{a:.3f}$\\pm${b:.3f}"
        lines.append(f"{m} & {pm(mean_c,std_c)} & {pm(arpd_m,arpd_s)} & {pm(hit_m,hit_s)} & {pm(worst_m,worst_s)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit("No input files matched.")

    per_seed_summaries = []
    for fp in files:
        df = pd.read_csv(fp)
        # basic validation
        missing = [c for c in ["instance","bks","seed"] + METHODS if c not in df.columns]
        if missing:
            raise SystemExit(f"{fp}: missing columns {missing}")
        # recompute gaps
        for m in METHODS:
            df[f"gap_{m}_pct"] = (df[m] - df["bks"]) / df["bks"] * 100.0
        seed = int(df["seed"].iloc[0])
        summ = compute_summary(df)
        summ["seed"] = seed
        out_fp = os.path.join(args.out_dir, f"summary_seed{seed}.csv")
        summ.to_csv(out_fp, index=False)
        per_seed_summaries.append(summ)

    all_s = pd.concat(per_seed_summaries, ignore_index=True)
    # aggregate across seeds
    agg = all_s.groupby("method").agg(["mean","std"])
    # flatten
    agg.columns = ["_".join([c,stat]) for c,stat in agg.columns]
    agg = agg.reset_index()
    agg_fp = os.path.join(args.out_dir, "summary_multiseed.csv")
    agg.to_csv(agg_fp, index=False)

    latex = make_latex_table(agg)
    tex_fp = os.path.join(args.out_dir, "table_main.tex")
    with open(tex_fp, "w", encoding="utf-8") as f:
        f.write(latex)

    print("Wrote:", agg_fp, tex_fp)

if __name__ == "__main__":
    main()
