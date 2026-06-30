#!/usr/bin/env python3
"""
RCPSP evaluation runner (NC/HC/LC_A/LC_B/LC_C) for PSPLIB datasets.

- Reads .sm instances
- Reads constraint JSONs (schema-compliant: instance,scores,edges,meta)
- Runs a small GA on activity permutations with an SSGS decoder
- Outputs per-seed CSVs: results_seed{seed}.csv

This is designed to be deterministic given:
  - seeds list
  - GA parameters
  - constraints JSONs

Example (Windows CMD/Anaconda Prompt):
  python run_rcpsp_experiment.py ^
    --sm_dir instances ^
    --constraints_dir out_local_qwen25 ^
    --out_dir results_qwen25 ^
    --bks_csv bks_j30.csv ^
    --seeds 0-29 ^
    --pop 16 --gen 15 ^
    --beta 0.5 --gamma 0.2 --lambda1 0.05 --lambda2 0.01 --hard_conf 0.9

Notes:
- Column "LC" is an alias for "LC_B" (main LC method).
"""

import argparse, os, re, json, random, zipfile, time, csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from array import array
import pandas as pd

@dataclass
class RCPSPInstance:
    name: str
    n_jobs: int
    horizon: int
    capacities: List[int]
    durations: List[int]
    demands: List[List[int]]
    successors: List[List[int]]
    predecessors: List[List[int]]


def _safe_extract_zip(zip_path: str, extract_dir: str) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Prevent path traversal.
            target = os.path.abspath(os.path.join(extract_dir, member))
            if not target.startswith(os.path.abspath(extract_dir) + os.sep) and target != os.path.abspath(extract_dir):
                raise ValueError(f"Unsafe zip member: {member}")
        zf.extractall(extract_dir)


def resolve_dataset_paths(args):
    """Resolve hierarchical project paths.

    Expected layout:
      data/<dataset>/<dataset>.zip
      data/<dataset>/bks_<dataset>.csv
      results/constraints/<dataset>/out_local_qwen25(.zip or extracted dir)
      results/ea_runs/<dataset>/
    Explicit CLI paths override inferred dataset paths.
    """
    project_root = os.path.abspath(args.project_root)
    dataset = args.dataset.lower() if args.dataset else None

    # Instance directory
    sm_dir = args.sm_dir
    if sm_dir is None:
        if not dataset:
            raise ValueError("Either --dataset or --sm_dir must be provided")
        data_dir = os.path.join(project_root, "data", dataset)
        preferred = os.path.join(data_dir, "instances")
        zip_path = os.path.join(data_dir, f"{dataset}.zip")
        if os.path.isdir(preferred) and any(f.endswith(".sm") for f in os.listdir(preferred)):
            sm_dir = preferred
        elif os.path.exists(zip_path):
            print(f"[INFO] extracting {zip_path} -> {preferred}")
            _safe_extract_zip(zip_path, preferred)
            sm_dir = preferred
        else:
            # fallback: maybe .sm files are placed directly under data/<dataset>
            if os.path.isdir(data_dir) and any(f.endswith(".sm") for f in os.listdir(data_dir)):
                sm_dir = data_dir
            else:
                raise FileNotFoundError(f"No instances found. Expected {zip_path} or {preferred}")

    # Constraint directory
    constraints_dir = args.constraints_dir
    if constraints_dir is None:
        if not dataset:
            raise ValueError("Either --dataset or --constraints_dir must be provided")
        cbase = os.path.join(project_root, "results", "constraints", dataset)
        preferred = os.path.join(cbase, args.constraints_name)
        zip_path = preferred + ".zip"
        if os.path.isdir(preferred) and any(f.endswith(".json") for f in os.listdir(preferred)):
            constraints_dir = preferred
        elif os.path.exists(zip_path):
            print(f"[INFO] extracting {zip_path} -> {preferred}")
            _safe_extract_zip(zip_path, preferred)
            constraints_dir = preferred
        elif os.path.isdir(cbase) and any(f.endswith(".json") for f in os.listdir(cbase)):
            constraints_dir = cbase
        else:
            raise FileNotFoundError(f"No constraints found. Expected {preferred} or {zip_path}")

    # Output directory
    out_dir = args.out_dir
    if out_dir is None:
        if not dataset:
            raise ValueError("Either --dataset or --out_dir must be provided")
        out_dir = os.path.join(project_root, "results", "ea_runs", dataset)

    # Reference CSV
    bks_csv = args.bks_csv
    if bks_csv is None:
        if not dataset:
            raise ValueError("Either --dataset or --bks_csv must be provided")
        bks_csv = os.path.join(project_root, "data", dataset, f"bks_{dataset}.csv")

    return sm_dir, constraints_dir, out_dir, bks_csv

def parse_sm(path: str) -> RCPSPInstance:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    name = os.path.basename(path)
    n_jobs = None
    horizon = None
    for line in lines:
        if line.strip().startswith("jobs"):
            m = re.search(r":\s*(\d+)", line)
            if m: n_jobs = int(m.group(1))
        if line.strip().startswith("horizon"):
            m = re.search(r":\s*(\d+)", line)
            if m: horizon = int(m.group(1))

    prec_start = req_start = avail_start = None
    for idx, line in enumerate(lines):
        if line.startswith("PRECEDENCE RELATIONS"): prec_start = idx
        if line.startswith("REQUESTS/DURATIONS"): req_start = idx
        if line.startswith("RESOURCEAVAILABILITIES"): avail_start = idx

    if n_jobs is None or horizon is None or prec_start is None or req_start is None or avail_start is None:
        raise ValueError(f"Failed to parse SM header/sections: {name}")

    succ = [[] for _ in range(n_jobs+1)]
    pred = [[] for _ in range(n_jobs+1)]

    # precedence
    i = prec_start
    while i < len(lines) and "jobnr." not in lines[i]:
        i += 1
    i += 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("*") or line.startswith("REQUESTS"):
            break
        if not line:
            i += 1
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            i += 1
            continue
        job = int(parts[0])
        n_succ = int(parts[2])
        succs = [int(x) for x in parts[3:3+n_succ]] if n_succ > 0 else []
        succ[job] = succs
        for s in succs:
            pred[s].append(job)
        i += 1

    # requests/durations
    i = req_start
    while i < len(lines) and "jobnr." not in lines[i]:
        i += 1
    header = lines[i].strip()
    R = header.count("R")
    i += 2

    durations = [0]*(n_jobs+1)
    demands = [[0]*R for _ in range(n_jobs+1)]
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("*") or line.startswith("RESOURCEAVAILABILITIES"):
            break
        if not line:
            i += 1
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            i += 1
            continue
        job = int(parts[0])
        duration = int(parts[2])
        res = [int(x) for x in parts[3:3+R]]
        durations[job] = duration
        demands[job] = res
        i += 1

    # capacities (next non-empty line after header)
    j = avail_start + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    j += 1  # typically skips the "R 1 R 2 ..." line
    while j < len(lines) and not lines[j].strip():
        j += 1
    capacities = [int(x) for x in lines[j].split()]

    return RCPSPInstance(
        name=name,
        n_jobs=n_jobs,
        horizon=horizon,
        capacities=capacities,
        durations=durations,
        demands=demands,
        successors=succ,
        predecessors=pred
    )

def ssgs_makespan(inst: RCPSPInstance, priority: List[int], select_rule=None) -> int:
    n = inst.n_jobs
    sink = n
    R = len(inst.capacities)
    max_dur = max(inst.durations)
    horizon = inst.horizon + max_dur + 1

    avail = [array('h', [cap]*horizon) for cap in inst.capacities]
    finish = [0]*(n+1)
    scheduled = [False]*(n+1)
    scheduled[1] = True

    remaining = set(priority)
    remaining.add(sink)
    order_list = priority + [sink]
    pos = {job: idx for idx, job in enumerate(priority)}
    pos[sink] = len(priority)

    while remaining:
        if select_rule is None:
            j = None
            for cand in order_list:
                if cand not in remaining:
                    continue
                ok = True
                for p in inst.predecessors[cand]:
                    if not scheduled[p]:
                        ok = False
                        break
                if ok:
                    j = cand
                    break
            if j is None:
                raise RuntimeError("No eligible job (decoder)")
        else:
            elig = []
            for cand in order_list:
                if cand not in remaining:
                    continue
                ok = True
                for p in inst.predecessors[cand]:
                    if not scheduled[p]:
                        ok = False
                        break
                if ok:
                    elig.append(cand)
            if not elig:
                raise RuntimeError("No eligible job (decoder)")
            j = select_rule(elig, pos, inst, remaining)

        # earliest precedence-ready
        ready = 0
        for p in inst.predecessors[j]:
            fp = finish[p]
            if fp > ready:
                ready = fp

        d = inst.durations[j]
        t = ready

        # resource feasibility search
        if d > 0:
            dem = inst.demands[j]
            while True:
                feasible = True
                for r in range(R):
                    req = dem[r]
                    if req == 0:
                        continue
                    arr = avail[r]
                    for tt in range(t, t+d):
                        if arr[tt] < req:
                            feasible = False
                            t = tt + 1
                            break
                    if not feasible:
                        break
                if feasible:
                    break

        finish[j] = t + d

        if d > 0:
            dem = inst.demands[j]
            for r in range(R):
                req = dem[r]
                if req == 0:
                    continue
                arr = avail[r]
                for tt in range(t, t+d):
                    arr[tt] -= req

        scheduled[j] = True
        remaining.remove(j)

    return finish[sink]

def random_perm(inst: RCPSPInstance, rng: random.Random) -> List[int]:
    jobs = list(range(2, inst.n_jobs))
    rng.shuffle(jobs)
    return jobs

def topo_order(inst: RCPSPInstance, rng: random.Random, priority: Optional[List[int]] = None,
               extra_edges: Optional[List[Tuple[int,int]]] = None) -> List[int]:
    n = inst.n_jobs
    jobs = list(range(2, n))
    job_set = set(jobs)

    adj = {j: [] for j in jobs}
    indeg = {j: 0 for j in jobs}

    for u in jobs:
        for v in inst.successors[u]:
            if v in job_set:
                adj[u].append(v)
                indeg[v] += 1

    if extra_edges:
        for (u, v) in extra_edges:
            if u in job_set and v in job_set:
                adj[u].append(v)
                indeg[v] += 1

    if priority is None:
        priority = jobs.copy()
        rng.shuffle(priority)

    pos = {j: i for i, j in enumerate(priority)}
    available = [j for j in jobs if indeg[j] == 0]
    order = []

    while available:
        available.sort(key=lambda j: pos.get(j, 10**9))
        j = available.pop(0)
        order.append(j)
        for v in adj[j]:
            indeg[v] -= 1
            if indeg[v] == 0:
                available.append(v)

    # if cycle (shouldn't happen), fall back
    if len(order) != len(jobs):
        return priority
    return order

def order_crossover(p1: List[int], p2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    n = len(p1)
    a = rng.randrange(n)
    b = rng.randrange(n)
    if a > b:
        a, b = b, a

    def ox(parent1, parent2):
        child = [None] * n
        child[a:b+1] = parent1[a:b+1]
        fill = [x for x in parent2 if x not in child[a:b+1]]
        idx = 0
        for i in list(range(0, a)) + list(range(b+1, n)):
            child[i] = fill[idx]
            idx += 1
        return child

    return ox(p1, p2), ox(p2, p1)

def swap_mutation(p: List[int], rng: random.Random) -> None:
    n = len(p)
    i = rng.randrange(n)
    j = rng.randrange(n)
    p[i], p[j] = p[j], p[i]

def edge_violation_penalty(order: List[int], edge_list: List[Tuple[int,int,float]]) -> float:
    pos = {job: i for i, job in enumerate(order)}
    pen = 0.0
    for u, v, conf in edge_list:
        if pos[u] > pos[v]:
            pen += conf
    return pen

def score_distance(order: List[int], pos_score: Dict[int,int]) -> int:
    pos = {job: i for i, job in enumerate(order)}
    return sum(abs(pos[j] - pos_score[j]) for j in order)

def make_select_rule_bias(scores: Dict[int,float],
                          out_edges: Dict[int,List[Tuple[int,float]]],
                          in_edges: Dict[int,List[Tuple[int,float]]],
                          beta: float, gamma: float):
    def rule(eligible: List[int], pos: Dict[int,int], inst: RCPSPInstance, remaining_set: set) -> int:
        best_j = None
        best_key = -1e18
        rem = remaining_set
        for j in eligible:
            key = -pos.get(j, 10**6) + beta * scores.get(j, 0.5)
            if gamma != 0.0:
                out_sum = sum(conf for (to, conf) in out_edges.get(j, []) if to in rem)
                in_sum = sum(conf for (fr, conf) in in_edges.get(j, []) if fr in rem)
                key += gamma * (out_sum - in_sum)
            if key > best_key:
                best_key = key
                best_j = j
        return best_j
    return rule

def run_ga(inst: RCPSPInstance, method: str, constraints: dict, rng: random.Random,
           pop_size=16, generations=15, cx_prob=0.9, mut_prob=0.2, elite=2,
           beta=0.5, gamma=0.2, lambda1=0.05, lambda2=0.01, hard_conf_threshold=0.9) -> int:
    jobs = list(range(2, inst.n_jobs))

    scores = {int(k): float(v) for k, v in (constraints.get("scores", {}) or {}).items()}
    edges = constraints.get("edges", []) or []

    hard_edges: List[Tuple[int,int]] = []
    soft_edges: List[Tuple[int,int,float]] = []
    for e in edges:
        u = int(e["i"]); v = int(e["j"])
        conf = float(e["conf"]); typ = e.get("type", "soft")
        if typ == "hard" or conf >= hard_conf_threshold:
            hard_edges.append((u, v))
        soft_edges.append((u, v, conf))

    score_items = [(j, scores.get(j, 0.5)) for j in jobs]
    score_items.sort(key=lambda x: (-x[1], x[0]))
    pos_score = {j: i for i, (j, _) in enumerate(score_items)}

    out_edges: Dict[int, List[Tuple[int,float]]] = {}
    in_edges: Dict[int, List[Tuple[int,float]]] = {}
    for u, v, conf in soft_edges:
        out_edges.setdefault(u, []).append((v, conf))
        in_edges.setdefault(v, []).append((u, conf))

    select_rule_bias = make_select_rule_bias(scores, out_edges, in_edges, beta, gamma) if method == "LC_B" else None

    def repair(order: List[int], extra: Optional[List[Tuple[int,int]]] = None) -> List[int]:
        return topo_order(inst, rng, priority=order, extra_edges=extra)

    # init
    pop: List[List[int]] = []
    if method == "HC":
        for _ in range(pop_size):
            pop.append(repair(random_perm(inst, rng), extra=None))
    elif method == "LC_A":
        for _ in range(pop_size):
            pop.append(repair(random_perm(inst, rng), extra=hard_edges))
    else:
        for _ in range(pop_size):
            pop.append(random_perm(inst, rng))

    def eval_ind(order: List[int]) -> Tuple[float, int]:
        if method == "LC_B":
            ms = ssgs_makespan(inst, order, select_rule=select_rule_bias)
        else:
            ms = ssgs_makespan(inst, order, select_rule=None)

        if method == "LC_C":
            pen_edges = edge_violation_penalty(order, soft_edges)
            dist = score_distance(order, pos_score)
            fit = ms + lambda1 * pen_edges + lambda2 * dist
        else:
            fit = ms
        return fit, ms

    fits: List[float] = []
    best_ms = 10**9
    for ind in pop:
        fit, ms = eval_ind(ind)
        fits.append(fit)
        if ms < best_ms:
            best_ms = ms

    def select_one() -> List[int]:
        best = None
        best_fit = None
        for _ in range(3):
            idx = rng.randrange(pop_size)
            f = fits[idx]
            if best is None or f < best_fit:
                best = pop[idx]
                best_fit = f
        return best  # type: ignore

    for _ in range(generations):
        elite_idx = sorted(range(pop_size), key=lambda i: fits[i])[:elite]
        new_pop = [pop[i][:] for i in elite_idx]
        while len(new_pop) < pop_size:
            p1 = select_one(); p2 = select_one()
            if rng.random() < cx_prob:
                c1, c2 = order_crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            if rng.random() < mut_prob:
                swap_mutation(c1, rng)
            if rng.random() < mut_prob:
                swap_mutation(c2, rng)

            if method == "HC":
                c1 = repair(c1, extra=None); c2 = repair(c2, extra=None)
            elif method == "LC_A":
                c1 = repair(c1, extra=hard_edges); c2 = repair(c2, extra=hard_edges)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop
        fits = []
        for ind in pop:
            fit, ms = eval_ind(ind)
            fits.append(fit)
            if ms < best_ms:
                best_ms = ms

    return best_ms

def parse_seeds(seed_str: str) -> List[int]:
    seed_str = seed_str.strip()
    if not seed_str:
        return []
    if "-" in seed_str:
        a, b = seed_str.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(seed_str)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="Project root containing data/ and results/")
    ap.add_argument("--dataset", default=None, choices=["j30", "j60", "j90", "j120"],
                    help="Dataset key. If provided, paths are inferred from the hierarchical project layout.")
    ap.add_argument("--sm_dir", default=None, help="Explicit directory containing .sm files. Overrides --dataset inference.")
    ap.add_argument("--constraints_dir", default=None, help="Explicit directory containing constraint JSONs. Overrides --dataset inference.")
    ap.add_argument("--constraints_name", default="out_local_qwen25",
                    help="Constraint folder name under results/constraints/<dataset>/")
    ap.add_argument("--out_dir", default=None, help="Explicit output directory. Default: results/ea_runs/<dataset>")
    ap.add_argument("--seeds", default="0-29", help="e.g., 0-29 or 0-9,10-19")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gen", type=int, default=15)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--lambda1", type=float, default=0.05)
    ap.add_argument("--lambda2", type=float, default=0.01)
    ap.add_argument("--hard_conf", type=float, default=0.9)
    ap.add_argument("--bks_csv", default=None, help="CSV with columns: instance,bks. Default: data/<dataset>/bks_<dataset>.csv")
    ap.add_argument("--runtime_log", default=None, help="CSV file for per-method EA runtime. Default: <out_dir>/ea_runtime_log.csv")
    args = ap.parse_args()

    args.sm_dir, args.constraints_dir, args.out_dir, args.bks_csv = resolve_dataset_paths(args)

    os.makedirs(args.out_dir, exist_ok=True)

    runtime_log = args.runtime_log or os.path.join(args.out_dir, "ea_runtime_log.csv")
    if not os.path.isabs(runtime_log):
        runtime_log = os.path.abspath(runtime_log)
    runtime_fields = [
        "dataset", "seed", "instance", "method", "elapsed_sec", "makespan",
        "pop", "gen", "nfe", "constraints_name"
    ]
    with open(runtime_log, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=runtime_fields).writeheader()

    def write_runtime_row(row):
        full = {k: row.get(k, "") for k in runtime_fields}
        with open(runtime_log, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=runtime_fields)
            writer.writerow(full)

    print(f"[INFO] sm_dir={args.sm_dir}")
    print(f"[INFO] constraints_dir={args.constraints_dir}")
    print(f"[INFO] out_dir={args.out_dir}")
    print(f"[INFO] bks_csv={args.bks_csv}")
    print(f"[INFO] runtime_log={runtime_log}")

    bks_df = pd.read_csv(args.bks_csv)
    if "instance" not in bks_df.columns or "bks" not in bks_df.columns:
        raise ValueError("bks_csv must contain columns: instance,bks")
    bks_map = dict(zip(bks_df["instance"], bks_df["bks"]))

    sm_files = sorted([os.path.join(args.sm_dir, f) for f in os.listdir(args.sm_dir) if f.endswith(".sm")])
    if not sm_files:
        raise ValueError(f"No .sm files found in {args.sm_dir}")
    instances = [parse_sm(p) for p in sm_files]

    constraints = {}
    for inst in instances:
        jpath = os.path.join(args.constraints_dir, inst.name.replace(".sm", ".json"))
        with open(jpath, "r", encoding="utf-8") as f:
            constraints[inst.name] = json.load(f)

    methods = ["NC", "HC", "LC_A", "LC_B", "LC_C"]
    method_offsets = {"NC": 0, "HC": 1000, "LC_A": 2000, "LC_B": 3000, "LC_C": 4000}

    seeds: List[int] = []
    for part in args.seeds.split(","):
        seeds.extend(parse_seeds(part))

    for seed in seeds:
        rows = []
        for idx, inst in enumerate(instances):
            base = seed * 100000 + idx
            cons = constraints[inst.name]
            bks = int(bks_map[inst.name])
            res = {}
            for m in methods:
                rng = random.Random(base + method_offsets[m])
                method_t0 = time.perf_counter()
                ms = run_ga(
                    inst, m, cons, rng,
                    pop_size=args.pop, generations=args.gen,
                    beta=args.beta, gamma=args.gamma,
                    lambda1=args.lambda1, lambda2=args.lambda2,
                    hard_conf_threshold=args.hard_conf,
                )
                elapsed = time.perf_counter() - method_t0
                write_runtime_row({
                    "dataset": args.dataset or "",
                    "seed": seed,
                    "instance": inst.name,
                    "method": m,
                    "elapsed_sec": f"{elapsed:.6f}",
                    "makespan": ms,
                    "pop": args.pop,
                    "gen": args.gen,
                    "nfe": args.pop * (args.gen + 1),
                    "constraints_name": args.constraints_name,
                })
                res[m] = ms

            row = {
                "instance": inst.name,
                "seed": seed,
                "bks": bks,
                "NC": res["NC"],
                "HC": res["HC"],
                "LC_A": res["LC_A"],
                "LC_B": res["LC_B"],
                "LC_C": res["LC_C"],
            }
            row["LC"] = row["LC_B"]
            for col in ["NC", "HC", "LC", "LC_A", "LC_B", "LC_C"]:
                row[f"gap_{col}_pct"] = (row[col] - bks) / bks * 100.0
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = os.path.join(args.out_dir, f"results_seed{seed}.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] seed={seed} -> {out_path}")

    # Summarize EA runtime by method.
    try:
        rt_df = pd.read_csv(runtime_log)
        if not rt_df.empty:
            summ = rt_df.groupby("method", as_index=False).agg(
                total_sec=("elapsed_sec", "sum"),
                mean_sec_per_instance_seed=("elapsed_sec", "mean"),
                n_calls=("elapsed_sec", "size"),
            )
            summ["dataset"] = args.dataset or ""
            summ["pop"] = args.pop
            summ["gen"] = args.gen
            summ["nfe"] = args.pop * (args.gen + 1)
            total_row = {
                "method": "EA_TOTAL",
                "total_sec": float(rt_df["elapsed_sec"].sum()),
                "mean_sec_per_instance_seed": float(rt_df["elapsed_sec"].mean()),
                "n_calls": int(len(rt_df)),
                "dataset": args.dataset or "",
                "pop": args.pop,
                "gen": args.gen,
                "nfe": args.pop * (args.gen + 1),
            }
            summ = pd.concat([summ, pd.DataFrame([total_row])], ignore_index=True)
            summary_path = os.path.join(args.out_dir, "ea_runtime_summary.csv")
            summ.to_csv(summary_path, index=False)
            print(f"[OK] EA runtime summary -> {summary_path}")
    except Exception as e:
        print(f"[WARN] failed to write EA runtime summary: {e}")

if __name__ == "__main__":
    main()
