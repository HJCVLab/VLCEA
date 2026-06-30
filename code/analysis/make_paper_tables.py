import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

METHODS = ["NC", "HC", "LC", "LC_A", "LC_B", "LC_C"]


def summarize_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        gaps = df[f"gap_{m}_pct"].to_numpy(dtype=float)
        ms = df[m].to_numpy(dtype=float)
        arpd = float(np.mean(gaps))
        opt_hits = int(np.sum(gaps == 0))
        rows.append(
            {
                "method": m,
                "mean_ms": float(np.mean(ms)),
                "ARPD_pct": arpd,
                "opt_hits": opt_hits,
                "opt_hit_rate": opt_hits / len(df),
            }
        )
    return pd.DataFrame(rows)


def latex_table_single(summary: pd.DataFrame, caption: str, label: str) -> str:
    order = METHODS
    df = summary.set_index("method").loc[order].reset_index()
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Method & Mean $C_{\\max}$ & ARPD (\\%) & Opt. hits & Opt. hit rate \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['method']} & {r['mean_ms']:.3f} & {r['ARPD_pct']:.3f} & {int(r['opt_hits'])} & {100*r['opt_hit_rate']:.2f}\\% \\\\" 
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def latex_table_multiseed(agg: pd.DataFrame, caption: str, label: str) -> str:
    order = METHODS
    df = agg.set_index("method").loc[order].reset_index()
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Method & Mean $C_{\\max}$ (\\,$\\pm$\\,) & ARPD (\\%) (\\,$\\pm$\\,) & Opt. hit rate (\\%) (\\,$\\pm$\\,) \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['method']} & {r['mean_ms']:.3f}$\\pm${r['std_ms']:.3f} & {r['mean_ARPD']:.3f}$\\pm${r['std_ARPD']:.3f} & {100*r['mean_opt_hit']:.2f}$\\pm${100*r['std_opt_hit']:.2f} \\\\" 
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help="e.g., 'results/seed*.csv'")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--single_seed", type=int, default=None, help="If set, also emit a single-seed table for that seed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit("No input files matched.")

    per_seed = []
    for p in paths:
        df = pd.read_csv(p)
        # try infer seed from filename
        base = os.path.basename(p)
        seed = None
        for tok in base.replace(".", "_").split("_"):
            if tok.isdigit():
                seed = int(tok)
                break
        sm = summarize_seed(df)
        sm["seed"] = seed if seed is not None else base
        per_seed.append(sm)

    all_sm = pd.concat(per_seed, ignore_index=True)
    all_sm.to_csv(os.path.join(args.out_dir, "seed_summaries.csv"), index=False)

    agg = (
        all_sm.groupby("method")
        .agg(
            mean_ARPD=("ARPD_pct", "mean"),
            std_ARPD=("ARPD_pct", "std"),
            mean_opt_hit=("opt_hit_rate", "mean"),
            std_opt_hit=("opt_hit_rate", "std"),
            mean_ms=("mean_ms", "mean"),
            std_ms=("mean_ms", "std"),
        )
        .reset_index()
    )
    agg.to_csv(os.path.join(args.out_dir, "multiseed_agg.csv"), index=False)

    tex = latex_table_multiseed(
        agg,
        caption="Multi-seed results (mean±std across seeds).",
        label="tab:multiseed",
    )
    with open(os.path.join(args.out_dir, "table_multiseed.tex"), "w") as f:
        f.write(tex)

    if args.single_seed is not None:
        # find a file containing that seed token
        candidates = [p for p in paths if str(args.single_seed) in os.path.basename(p)]
        if candidates:
            df = pd.read_csv(candidates[0])
            sm = summarize_seed(df)
            tex = latex_table_single(
                sm,
                caption=f"Single-seed (seed={args.single_seed}) results.",
                label="tab:single_seed",
            )
            with open(os.path.join(args.out_dir, f"table_seed{args.single_seed}.tex"), "w") as f:
                f.write(tex)


if __name__ == "__main__":
    main()
