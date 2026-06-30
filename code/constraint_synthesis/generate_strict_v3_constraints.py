#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic strict-v3 constraint generator for PSPLIB RCPSP .sm instances.

Inputs:
  - a zip file containing *.sm (e.g., j30.sm.zip), OR
  - a folder containing *.sm

Outputs:
  - one JSON per instance in out_dir

This matches the "rcpsp-mixed-strict-v3" recipe:
  scores: intensity/duration/outdegree min-max normalization + weighted sum
  edges: direct precedence edges only, delta thresholds for hard/soft, caps
"""

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def parse_n_jobs(text: str) -> Optional[int]:
    m = re.search(r"jobs\s*\(incl\.\s*supersource/sink\s*\)\s*:\s*(\d+)", text, re.I)
    return int(m.group(1)) if m else None


def parse_num_renewable(text: str) -> Optional[int]:
    m = re.search(r"renewable\s*:\s*(\d+)\s*R", text, re.I)
    return int(m.group(1)) if m else None


def extract_block(text: str, start_pat: str, end_pat: str) -> Optional[str]:
    s = re.search(start_pat, text, re.I)
    if not s:
        return None
    e = re.search(end_pat, text[s.end():], re.I)
    if not e:
        return None
    return text[s.end(): s.end() + e.start()]


def parse_capacities(text: str) -> Optional[List[int]]:
    m = re.search(r"RESOURCEAVAILABILITIES\s*:", text, re.I)
    if not m:
        return None
    after = text[m.end():]
    lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
    int_lines = []
    for ln in lines:
        nums = re.findall(r"-?\d+", ln)
        if len(nums) >= 2:
            int_lines.append([int(x) for x in nums])
    if not int_lines:
        return None
    cand = int_lines[-1]
    k = parse_num_renewable(text)
    if k is not None and len(cand) >= k:
        return cand[:k]
    return cand


def parse_precedence_successors(text: str) -> Optional[Dict[int, List[int]]]:
    block = extract_block(text, r"PRECEDENCE\s+RELATIONS\s*:", r"REQUESTS/DURATIONS\s*:")
    if block is None:
        return None
    succ: Dict[int, List[int]] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if "jobnr" in line.lower() or "successors" in line.lower() or "modes" in line.lower():
            continue
        nums = re.findall(r"-?\d+", line)
        if len(nums) < 3:
            continue
        job = int(nums[0])
        n_succ = int(nums[2])
        succs = [int(x) for x in nums[3:3 + n_succ]] if n_succ > 0 else []
        succ[job] = succs
    return succ if succ else None


def parse_requests_durations(text: str) -> Optional[Tuple[Dict[int, int], Dict[int, List[int]]]]:
    block = extract_block(text, r"REQUESTS/DURATIONS\s*:", r"RESOURCEAVAILABILITIES\s*:")
    if block is None:
        return None
    k = parse_num_renewable(text)
    if k is None:
        return None
    duration: Dict[int, int] = {}
    req: Dict[int, List[int]] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if "jobnr" in line.lower() or "duration" in line.lower():
            continue
        if set(line) == {"-"}:
            continue
        nums = re.findall(r"-?\d+", line)
        if len(nums) < 3 + k:
            continue
        j = int(nums[0])
        d = int(nums[2])
        demands = [int(x) for x in nums[3:3 + k]]
        duration[j] = d
        req[j] = demands
    return (duration, req) if duration and req else None


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def round3(x: float) -> float:
    return float(f"{x:.3f}")


def minmax_norm(values: Dict[int, float], jobs: List[int]) -> Dict[int, float]:
    xs = [values[j] for j in jobs]
    mn = min(xs)
    mx = max(xs)
    if abs(mx - mn) < 1e-12:
        return {j: 0.5 for j in jobs}
    return {j: (values[j] - mn) / (mx - mn) for j in jobs}


def compute_constraints(text: str, instance_name: str, llm_id: str) -> Dict[str, Any]:
    n = parse_n_jobs(text)
    caps = parse_capacities(text)
    succ = parse_precedence_successors(text)
    rd = parse_requests_durations(text)

    if n is None or caps is None or succ is None or rd is None:
        scores = {}
        if n is not None:
            for j in range(2, n):
                scores[str(j)] = 0.500
        return {
            "instance": instance_name,
            "scores": scores,
            "edges": [],
            "meta": {"llm": llm_id, "prompt_version": "rcpsp-mixed-strict-v3"},
        }

    duration, req = rd
    k = len(caps)
    real = list(range(2, n))

    intensity: Dict[int, float] = {}
    outdeg: Dict[int, float] = {}
    durf: Dict[int, float] = {}

    for j in real:
        durf[j] = float(duration.get(j, 0))
        demands = req.get(j, [0] * k)
        inten = 0.0
        for kk in range(k):
            cap = caps[kk]
            if cap == 0:
                continue
            inten += float(demands[kk]) / float(cap)
        intensity[j] = inten
        outdeg[j] = float(len(succ.get(j, [])))

    d_norm = minmax_norm(durf, real)
    i_norm = minmax_norm(intensity, real)
    o_norm = minmax_norm(outdeg, real)

    scores: Dict[str, float] = {}
    for j in real:
        raw = 0.50 * i_norm[j] + 0.30 * d_norm[j] + 0.20 * o_norm[j]
        scores[str(j)] = round3(clip(raw, 0.0, 1.0))

    # edges
    cand = []
    for i in real:
        for j in succ.get(i, []):
            if 2 <= j <= n - 1:
                delta = scores[str(i)] - scores[str(j)]
                cand.append((delta, i, j))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))

    edges = []
    used = set()

    # hard edges
    hard_count = 0
    for delta, i, j in cand:
        if hard_count >= 5:
            break
        if delta >= 0.150 and (i, j) not in used:
            edges.append({"i": i, "j": j, "conf": 0.950, "type": "hard"})
            used.add((i, j))
            hard_count += 1

    # soft edges
    for delta, i, j in cand:
        if len(edges) >= 30:
            break
        if (i, j) in used:
            continue
        if delta < 0.050:
            continue
        conf = round3(clip(0.60 + 0.30 * delta, 0.60, 0.90))
        edges.append({"i": i, "j": j, "conf": conf, "type": "soft"})
        used.add((i, j))

    # sort edges: hard first, then delta desc
    def edge_key(e):
        t_rank = 0 if e["type"] == "hard" else 1
        d = scores.get(str(e["i"]), 0.5) - scores.get(str(e["j"]), 0.5)
        return (t_rank, -d, e["i"], e["j"])

    edges = sorted(edges, key=edge_key)

    scores_sorted = {k: scores[k] for k in sorted(scores.keys(), key=lambda x: int(x))}

    return {
        "instance": instance_name,
        "scores": scores_sorted,
        "edges": edges,
        "meta": {"llm": llm_id, "prompt_version": "rcpsp-mixed-strict-v3"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm_zip", type=str, default="", help="Path to zip containing *.sm")
    ap.add_argument("--sm_dir", type=str, default="", help="Path to directory containing *.sm")
    ap.add_argument("--out_dir", type=str, default="out_constraints_strict_v3")
    ap.add_argument("--llm_id", type=str, default="python/deterministic-strict-v3")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances: List[Tuple[str, str]] = []

    if args.sm_zip:
        zf = zipfile.ZipFile(args.sm_zip)
        for name in sorted(zf.namelist()):
            if name.lower().endswith(".sm"):
                text = zf.read(name).decode("utf-8", errors="replace")
                instances.append((name, text))
    elif args.sm_dir:
        sm_dir = Path(args.sm_dir)
        for p in sorted(sm_dir.glob("*.sm")):
            instances.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
    else:
        raise SystemExit("Provide --sm_zip or --sm_dir")

    for name, text in instances:
        obj = compute_constraints(text, name, args.llm_id)
        out_path = out_dir / (Path(name).stem + ".json")
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: wrote {len(instances)} JSON files to {out_dir}")


if __name__ == "__main__":
    main()
