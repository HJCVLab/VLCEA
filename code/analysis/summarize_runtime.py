#!/usr/bin/env python3
"""Summarize VLCEA wall-clock/runtime logs for one dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project_root', default='.')
    ap.add_argument('--dataset', required=True, choices=['j30','j60','j90','j120'])
    ap.add_argument('--constraints_name', default='out_local_qwen25')
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    root = Path(args.project_root).resolve(); ds = args.dataset.lower()
    out_dir = Path(args.out_dir) if args.out_dir else root / 'results' / 'runtime'
    if not out_dir.is_absolute(): out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_log = root / 'results' / 'constraints' / ds / args.constraints_name / 'generation_time_log.csv'
    ea_log = root / 'results' / 'ea_runs' / ds / 'ea_runtime_log.csv'
    rows = []
    if gen_log.exists():
        g = pd.read_csv(gen_log)
        g['elapsed_sec'] = pd.to_numeric(g['elapsed_sec'], errors='coerce').fillna(0.0)
        n = int(len(g)); total = float(g['elapsed_sec'].sum())
        fallback_count = int(g['fallback'].astype(str).str.lower().isin(['true','1','yes']).sum()) if 'fallback' in g.columns else 0
        mean_edges = float(pd.to_numeric(g['edges'], errors='coerce').mean()) if 'edges' in g.columns else float('nan')
        mean_hard = float(pd.to_numeric(g['hard_edges'], errors='coerce').mean()) if 'hard_edges' in g.columns else float('nan')
        mean_attempts = float(pd.to_numeric(g['attempts_used'], errors='coerce').mean()) if 'attempts_used' in g.columns else float('nan')
        pd.DataFrame([{
            'dataset': ds, 'constraints_name': args.constraints_name, 'n_instances': n,
            'fallback_count': fallback_count, 'fallback_rate_pct': fallback_count / n * 100.0 if n else float('nan'),
            'total_generation_sec': total, 'mean_generation_sec_per_instance': total / n if n else float('nan'),
            'mean_attempts_used': mean_attempts, 'mean_edges': mean_edges, 'mean_hard_edges': mean_hard,
        }]).to_csv(out_dir / f'{ds}_constraint_generation_summary.csv', index=False)
        rows.append({'dataset': ds, 'component': 'constraint_generation', 'total_sec': total, 'n_units': n,
                     'mean_sec_per_unit': total / n if n else float('nan'),
                     'amortized_sec_per_instance_seed': total / (n * args.seeds) if n and args.seeds else float('nan'),
                     'notes': f'one-time LLM preprocessing; amortized over {args.seeds} EA seeds'})
    else:
        print(f'[WARN] missing generation log: {gen_log}')
    if ea_log.exists():
        e = pd.read_csv(ea_log)
        e['elapsed_sec'] = pd.to_numeric(e['elapsed_sec'], errors='coerce').fillna(0.0)
        m = e.groupby('method', as_index=False).agg(total_sec=('elapsed_sec','sum'), mean_sec_per_instance_seed=('elapsed_sec','mean'), n_calls=('elapsed_sec','size'))
        m.insert(0, 'dataset', ds)
        m.to_csv(out_dir / f'{ds}_ea_method_runtime_summary.csv', index=False)
        total = float(e['elapsed_sec'].sum())
        rows.append({'dataset': ds, 'component': 'ea_total_all_methods', 'total_sec': total, 'n_units': int(len(e)),
                     'mean_sec_per_unit': float(e['elapsed_sec'].mean()) if len(e) else float('nan'),
                     'amortized_sec_per_instance_seed': '', 'notes': 'sum over all methods, instances, and seeds'})
        for _, r in m.iterrows():
            rows.append({'dataset': ds, 'component': f"ea_{r['method']}", 'total_sec': float(r['total_sec']),
                         'n_units': int(r['n_calls']), 'mean_sec_per_unit': float(r['mean_sec_per_instance_seed']),
                         'amortized_sec_per_instance_seed': '', 'notes': 'per method over all instances and seeds'})
    else:
        print(f'[WARN] missing EA runtime log: {ea_log}')
    out_path = out_dir / f'{ds}_runtime_summary.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'[OK] runtime summary -> {out_path}')

if __name__ == '__main__':
    main()
