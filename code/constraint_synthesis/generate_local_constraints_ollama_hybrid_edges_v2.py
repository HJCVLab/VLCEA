# -*- coding: utf-8 -*-
"""
RCPSP Mixed-Constraint JSON generator using local Ollama.
HYBRID MODE (recommended for strict-v3 stability):
  - LLM is asked to output ONLY scores (still wrapped in the full schema object).
  - edges are ALWAYS (re)computed in Python using strict-v3 edge recipe from:
        (scores + direct precedence relations)
  - This avoids the common issue where local LLMs omit edges or invent edges.

Key fixes vs earlier versions:
  1) Per-instance "format" schema is generated to REQUIRE ALL score keys "2".."N-1".
     -> prevents the "Missing scores for 1 activity" problem.
  2) Output token budget increased controllably via --num_predict.
  3) Optional fallback to deterministic strict-v3 score computation if LLM still fails.

Requires:
  pip install requests jsonschema

Run (Windows CMD / Anaconda Prompt):
  python generate_local_constraints_ollama_hybrid_edges_v2.py ^
    --prompt_template prompt_template_scores_only_v3.txt ^
    --schema schema.json ^
    --instances_dir instances ^
    --out_dir out_local_qwen25 ^
    --out_raw_dir out_local_qwen25_raw ^
    --model qwen2.5:7b-instruct ^
    --llm_id "ollama/qwen2.5:7b-instruct" ^
    --temperature 0 ^
    --seed 0 ^
    --num_ctx 8192 ^
    --num_predict 4096 ^
    --max_attempts 6 ^
    --limit 3
"""

import argparse
import csv
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from jsonschema import Draft7Validator


# -----------------------------
# IO / template
# -----------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_system_user(template: str) -> Tuple[str, str]:
    if "SYSTEM:" in template and "USER:" in template:
        sys_idx = template.find("SYSTEM:")
        usr_idx = template.find("USER:")
        system = template[sys_idx + len("SYSTEM:"):usr_idx].strip()
        user = template[usr_idx + len("USER:"):].strip()
        return system, user
    return "", template.strip()


def render_template(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


# -----------------------------
# Parse .sm/.bas (minimal for schema + edges)
# -----------------------------

def parse_n_jobs(text: str) -> Optional[int]:
    m = re.search(r"jobs\s*\(incl\.\s*supersource/sink\s*\)\s*:\s*(\d+)", text, re.I)
    return int(m.group(1)) if m else None


def extract_block(text: str, start_pat: str, end_pat: str) -> Optional[str]:
    s = re.search(start_pat, text, flags=re.IGNORECASE)
    if not s:
        return None
    e = re.search(end_pat, text[s.end():], flags=re.IGNORECASE)
    if not e:
        return None
    return text[s.end(): s.end() + e.start()]


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
        try:
            job = int(nums[0])
            n_succ = int(nums[2])
            succs = [int(x) for x in nums[3:3 + n_succ]] if n_succ > 0 else []
            succ[job] = succs
        except Exception:
            return None
    return succ if succ else None


# -----------------------------
# Strict-v3 edge builder (from scores + direct precedence)
# -----------------------------

MAX_EDGES_TOTAL = 30
MAX_HARD_EDGES = 5
HARD_DELTA_MIN = 0.150
SOFT_DELTA_MIN = 0.050

def round3(x: float) -> float:
    return float(f"{x:.3f}")

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def build_edges_from_scores(sm_text: str, scores: Dict[str, float], max_edges_total: int = MAX_EDGES_TOTAL, max_hard_edges: int = MAX_HARD_EDGES, hard_delta_min: float = HARD_DELTA_MIN, soft_delta_min: float = SOFT_DELTA_MIN) -> List[Dict[str, Any]]:
    n = parse_n_jobs(sm_text)
    succ = parse_precedence_successors(sm_text)
    if n is None or succ is None:
        return []  # strict rule: if precedence parsing fails, edges=[]

    def s(j: int) -> float:
        return float(scores.get(str(j), 0.5))

    real = range(2, n)  # 2..N-1
    cand = []
    for i in real:
        for j in succ.get(i, []):
            if 2 <= j <= n - 1:
                delta = s(i) - s(j)
                cand.append((delta, i, j))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))

    edges: List[Dict[str, Any]] = []
    used = set()

    # hard edges
    for delta, i, j in cand:
        if len([e for e in edges if e["type"] == "hard"]) >= max_hard_edges:
            break
        if delta >= hard_delta_min and (i, j) not in used:
            edges.append({"i": i, "j": j, "conf": 0.950, "type": "hard"})
            used.add((i, j))

    # soft edges
    for delta, i, j in cand:
        if len(edges) >= max_edges_total:
            break
        if (i, j) in used:
            continue
        if delta < soft_delta_min:
            continue
        conf = round3(clip(0.60 + 0.30 * delta, 0.60, 0.90))
        edges.append({"i": i, "j": j, "conf": conf, "type": "soft"})
        used.add((i, j))

    # sort hard first, then by delta desc
    edges.sort(key=lambda e: (0 if e["type"] == "hard" else 1,
                              -(s(int(e["i"])) - s(int(e["j"]))),
                              int(e["i"]), int(e["j"])))
    return edges


# -----------------------------
# Deterministic strict-v3 score fallback (optional)
# -----------------------------

def parse_num_renewable(sm_text: str) -> Optional[int]:
    m = re.search(r"renewable\s*:\s*(\d+)\s*R", sm_text, re.I)
    return int(m.group(1)) if m else None

def parse_capacities(sm_text: str) -> Optional[List[int]]:
    block = extract_block(sm_text, r"RESOURCEAVAILABILITIES\s*:", r"\*{10,}")
    if block is None:
        m = re.search(r"RESOURCEAVAILABILITIES\s*:", sm_text, re.I)
        if not m:
            return None
        block = sm_text[m.end():]
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    int_lines = []
    for ln in lines:
        nums = re.findall(r"-?\d+", ln)
        if len(nums) >= 2:
            int_lines.append([int(x) for x in nums])
    if not int_lines:
        return None
    k = parse_num_renewable(sm_text)
    cand = int_lines[-1]
    if k is not None and len(cand) >= k:
        return cand[:k]
    return cand

def parse_requests_durations(sm_text: str) -> Optional[Tuple[Dict[int, int], Dict[int, List[int]]]]:
    block = extract_block(sm_text, r"REQUESTS/DURATIONS\s*:", r"RESOURCEAVAILABILITIES\s*:")
    if block is None:
        return None
    k = parse_num_renewable(sm_text)
    if k is None:
        return None
    duration: Dict[int, int] = {}
    req: Dict[int, List[int]] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if "jobnr" in line.lower() or "duration" in line.lower() or set(line) == {"-"}:
            continue
        nums = re.findall(r"-?\d+", line)
        if len(nums) < 3 + k:
            continue
        try:
            j = int(nums[0]); d = int(nums[2])
            demands = [int(x) for x in nums[3:3 + k]]
            duration[j] = d
            req[j] = demands
        except Exception:
            return None
    return (duration, req) if duration and req else None

def minmax_norm(values: Dict[int, float], jobs: List[int]) -> Dict[int, float]:
    xs = [values[j] for j in jobs]
    mn = min(xs); mx = max(xs)
    out: Dict[int, float] = {}
    if abs(mx - mn) < 1e-12:
        for j in jobs:
            out[j] = 0.5
        return out
    for j in jobs:
        out[j] = (values[j] - mn) / (mx - mn)
    return out

def compute_strict_v3_scores(sm_text: str) -> Optional[Dict[str, float]]:
    n = parse_n_jobs(sm_text)
    caps = parse_capacities(sm_text)
    succ = parse_precedence_successors(sm_text)
    rd = parse_requests_durations(sm_text)
    if n is None or caps is None or succ is None or rd is None:
        return None
    duration, req = rd
    k = len(caps)
    real = list(range(2, n))

    dur_f: Dict[int, float] = {}
    inten: Dict[int, float] = {}
    outdeg: Dict[int, float] = {}

    for j in real:
        dur_f[j] = float(duration.get(j, 0))
        demands = req.get(j, [0] * k)
        x = 0.0
        for kk in range(k):
            cap = caps[kk]
            if cap == 0:
                continue
            x += float(demands[kk]) / float(cap)
        inten[j] = x
        outdeg[j] = float(len(succ.get(j, [])))

    d_norm = minmax_norm(dur_f, real)
    i_norm = minmax_norm(inten, real)
    o_norm = minmax_norm(outdeg, real)

    scores: Dict[str, float] = {}
    for j in real:
        raw = 0.50 * i_norm[j] + 0.30 * d_norm[j] + 0.20 * o_norm[j]
        scores[str(j)] = round3(clip(raw, 0.0, 1.0))
    return scores


# -----------------------------
# Ollama structured output schema (per instance)
# -----------------------------

def make_instance_format_schema(base_schema: Dict[str, Any], n: int, force_edges_empty: bool = True) -> Dict[str, Any]:
    """
    Create a stricter schema for Ollama `format` so the model MUST output all score keys 2..N-1.
    This is only for generation-time control. Final JSON is validated against the original schema.json.
    """
    # base outer schema
    s = dict(base_schema)
    s.pop("$schema", None)
    s.pop("title", None)

    # Build required score properties
    score_props = {}
    required_keys = [str(j) for j in range(2, n)]
    for k in required_keys:
        score_props[k] = {"type": "number", "minimum": 0, "maximum": 1}

    s["properties"] = dict(s.get("properties", {}))
    s["properties"]["scores"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": score_props,
        "required": required_keys,
    }

    # Encourage minimal edges output (we overwrite edges anyway)
    if force_edges_empty:
        s["properties"]["edges"] = {
            "type": "array",
            "maxItems": 0
        }

    return s


# -----------------------------
# Ollama call
# -----------------------------

def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    format_schema: Dict[str, Any],
    temperature: float,
    seed: int,
    num_ctx: int,
    num_predict: int,
    timeout_sec: float,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": format_schema,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
    }
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    return json.loads(text)  # with structured output, should be pure JSON


# -----------------------------
# Canonicalize / validate
# -----------------------------

def canonicalize_meta(obj: Dict[str, Any], instance_name: str, llm_id: str, timestamp: Optional[str]) -> Dict[str, Any]:
    obj["instance"] = instance_name
    meta = obj.get("meta") or {}
    meta["llm"] = llm_id
    meta["prompt_version"] = "rcpsp-mixed-strict-v3"
    if timestamp:
        meta["timestamp"] = timestamp
    else:
        meta.pop("timestamp", None)
    obj["meta"] = meta
    return obj


def required_score_keys(n: int) -> List[str]:
    return [str(j) for j in range(2, n)]


def check_scores_complete_and_nontrivial(scores: Dict[str, Any], n: int) -> Tuple[bool, str]:
    req = required_score_keys(n)
    missing = [k for k in req if k not in scores]
    extra = [k for k in scores.keys() if k not in req]
    if missing:
        return False, f"Missing scores for keys: {missing}"
    if extra:
        return False, f"Extra score keys not allowed: {extra[:10]}"
    vals = [float(scores[k]) for k in req]
    if max(vals) - min(vals) < 1e-9:
        return False, "All scores are identical (FAIL-SAFE-like)."
    return True, ""


def make_failsafe_json(instance_name: str, llm_id: str, n: Optional[int]) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    if n is not None and n >= 3:
        for j in range(2, n):
            scores[str(j)] = 0.500
    return {
        "instance": instance_name,
        "scores": scores,
        "edges": [],
        "meta": {"llm": llm_id, "prompt_version": "rcpsp-mixed-strict-v3"}
    }


# -----------------------------
# Path resolution
# -----------------------------


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        root = extract_dir.resolve()
        for member in zf.namelist():
            target = (extract_dir / member).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"Unsafe zip member: {member}")
        zf.extractall(extract_dir)


def resolve_generation_paths(args) -> Tuple[Path, Path, Path, Path, Path]:
    """Resolve paths under the hierarchical project layout.

    Expected layout:
      data/<dataset>/<dataset>.zip
      results/constraints/<dataset>/<constraints_name>/
      results/constraints/<dataset>/<constraints_name>_raw/
    Explicit paths override dataset inference.
    """
    project_root = Path(args.project_root).resolve()
    dataset = args.dataset.lower() if args.dataset else None

    prompt_path = Path(args.prompt_template)
    schema_path = Path(args.schema)
    if not prompt_path.is_absolute():
        prompt_path = project_root / prompt_path
    if not schema_path.is_absolute():
        schema_path = project_root / schema_path

    if args.instances_dir:
        inst_dir = Path(args.instances_dir)
        if not inst_dir.is_absolute():
            inst_dir = project_root / inst_dir
    else:
        if not dataset:
            raise ValueError("Either --dataset or --instances_dir must be provided")
        data_dir = project_root / "data" / dataset
        inst_dir = data_dir / "instances"
        zip_path = data_dir / f"{dataset}.zip"
        if inst_dir.exists() and list(inst_dir.glob("*.sm")):
            pass
        elif zip_path.exists():
            print(f"[INFO] extracting {zip_path} -> {inst_dir}")
            _safe_extract_zip(zip_path, inst_dir)
        elif data_dir.exists() and list(data_dir.glob("*.sm")):
            inst_dir = data_dir
        else:
            raise FileNotFoundError(f"No instances found. Expected {zip_path} or {inst_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
    else:
        if not dataset:
            raise ValueError("Either --dataset or --out_dir must be provided")
        out_dir = project_root / "results" / "constraints" / dataset / args.constraints_name

    if args.out_raw_dir:
        raw_dir = Path(args.out_raw_dir)
        if not raw_dir.is_absolute():
            raw_dir = project_root / raw_dir
    else:
        if not dataset:
            raise ValueError("Either --dataset or --out_raw_dir must be provided")
        raw_dir = project_root / "results" / "constraints" / dataset / f"{args.constraints_name}_raw"

    return prompt_path, schema_path, inst_dir, out_dir, raw_dir

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="Project root containing data/ and results/")
    ap.add_argument("--dataset", default=None, choices=["j30", "j60", "j90", "j120"], help="Dataset key. If provided, paths are inferred from the hierarchical layout.")
    ap.add_argument("--constraints_name", default="out_local_qwen25", help="Output constraint folder under results/constraints/<dataset>/")
    ap.add_argument("--prompt_template", default="code/constraint_synthesis/prompt_template_scores_only_v3_min.txt")
    ap.add_argument("--schema", default="code/constraint_synthesis/rcpsp_mixed_constraint_schema.json")
    ap.add_argument("--instances_dir", default=None, help="Explicit directory containing .sm files. Overrides --dataset inference.")
    ap.add_argument("--out_dir", default=None, help="Explicit output directory. Default: results/constraints/<dataset>/<constraints_name>")
    ap.add_argument("--out_raw_dir", default=None, help="Explicit raw-output directory. Default: results/constraints/<dataset>/<constraints_name>_raw")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm_id", default="", help='meta.llm. empty => "ollama/<model>"')
    ap.add_argument("--timestamp", default="", help="empty => omit meta.timestamp")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--num_predict", type=int, default=4096)
    ap.add_argument("--timeout_sec", type=float, default=600.0)
    ap.add_argument("--max_attempts", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--inject_schema_in_prompt", action="store_true",
                    help="Not recommended. Schema is enforced via Ollama 'format' already.")
    ap.add_argument("--fallback_det_scores", action="store_true",
                    help="If LLM fails after retries, compute strict-v3 scores in Python (then edges) instead of FAIL-SAFE 0.5.")
    ap.add_argument("--max_edges_total", type=int, default=MAX_EDGES_TOTAL)
    ap.add_argument("--max_hard_edges", type=int, default=MAX_HARD_EDGES)
    ap.add_argument("--hard_delta_min", type=float, default=HARD_DELTA_MIN)
    ap.add_argument("--soft_delta_min", type=float, default=SOFT_DELTA_MIN)
    ap.add_argument("--runtime_log", default=None,
                    help="CSV file for per-instance constraint generation runtime. Default: <out_dir>/generation_time_log.csv")
    args = ap.parse_args()

    prompt_path, schema_path, inst_dir, out_dir, raw_dir = resolve_generation_paths(args)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(args.project_root).resolve()
    runtime_log = Path(args.runtime_log) if args.runtime_log else (out_dir / "generation_time_log.csv")
    if not runtime_log.is_absolute():
        runtime_log = project_root / runtime_log
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_fields = [
        "dataset", "instance", "status", "fallback", "attempts_used",
        "elapsed_sec", "score_count", "score_min", "score_max", "unique_scores",
        "edges", "hard_edges", "soft_edges", "error", "output_json"
    ]
    with runtime_log.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=runtime_fields).writeheader()

    def write_runtime_row(row: Dict[str, Any]) -> None:
        full = {k: row.get(k, "") for k in runtime_fields}
        with runtime_log.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=runtime_fields)
            writer.writerow(full)

    def score_stats(scores_obj: Dict[str, Any]) -> Tuple[int, Any, Any, int]:
        if not scores_obj:
            return 0, "", "", 0
        vals = [float(v) for v in scores_obj.values()]
        return len(vals), min(vals), max(vals), len(set(vals))

    def edge_stats(edges_obj: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        total = len(edges_obj or [])
        hard = sum(1 for e in (edges_obj or []) if e.get("type") == "hard")
        return total, hard, total - hard

    if not prompt_path.exists():
        print(f"[ERROR] missing prompt_template: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    if not schema_path.exists():
        print(f"[ERROR] missing schema: {schema_path}", file=sys.stderr)
        sys.exit(1)
    if not inst_dir.exists():
        print(f"[ERROR] missing instances_dir: {inst_dir}", file=sys.stderr)
        sys.exit(1)

    llm_id = args.llm_id.strip() or f"ollama/{args.model}"
    timestamp = args.timestamp.strip() or None

    template = read_text(prompt_path)
    system_tmpl, user_tmpl = split_system_user(template)

    base_schema = json.loads(read_text(schema_path))
    base_validator = Draft7Validator(base_schema)

    files = sorted(inst_dir.glob("*.sm"))
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"[ERROR] no *.sm files in {inst_dir}", file=sys.stderr)
        sys.exit(1)

    failures: List[str] = []

    print(f"[INFO] instances: {len(files)}")
    print(f"[INFO] model: {args.model} (meta.llm={llm_id})")
    print(f"[INFO] ollama_url: {args.ollama_url}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] runtime_log: {runtime_log}")
    print(f"[INFO] num_ctx={args.num_ctx}, num_predict={args.num_predict}, seed={args.seed}, temperature={args.temperature}")
    print(f"[INFO] edge_budget: M={args.max_edges_total}, M_h={args.max_hard_edges}, hard_delta_min={args.hard_delta_min}, soft_delta_min={args.soft_delta_min}")

    for idx, sm_path in enumerate(files, 1):
        instance_name = sm_path.name
        out_path = out_dir / (sm_path.stem + ".json")

        inst_t0 = time.perf_counter()
        if out_path.exists() and not args.overwrite:
            sc_n = sc_min = sc_max = sc_unique = ""
            e_total = e_hard = e_soft = ""
            try:
                prev = json.loads(out_path.read_text(encoding="utf-8", errors="replace"))
                sc_n, sc_min, sc_max, sc_unique = score_stats(prev.get("scores") or {})
                e_total, e_hard, e_soft = edge_stats(prev.get("edges") or [])
            except Exception:
                pass
            write_runtime_row({
                "dataset": args.dataset or "",
                "instance": instance_name,
                "status": "skipped_exists",
                "fallback": False,
                "attempts_used": 0,
                "elapsed_sec": f"{time.perf_counter() - inst_t0:.6f}",
                "score_count": sc_n, "score_min": sc_min, "score_max": sc_max, "unique_scores": sc_unique,
                "edges": e_total, "hard_edges": e_hard, "soft_edges": e_soft,
                "error": "",
                "output_json": str(out_path),
            })
            print(f"[SKIP] ({idx}/{len(files)}) {instance_name} -> exists")
            continue

        sm_text = read_text(sm_path)
        n = parse_n_jobs(sm_text)
        if n is None:
            fs = make_failsafe_json(instance_name, llm_id, None)
            out_path.write_text(json.dumps(fs, ensure_ascii=False, indent=2), encoding="utf-8")
            failures.append(instance_name + " (cannot parse N)")
            write_runtime_row({
                "dataset": args.dataset or "",
                "instance": instance_name,
                "status": "failsafe_parse_n",
                "fallback": True,
                "attempts_used": 0,
                "elapsed_sec": f"{time.perf_counter() - inst_t0:.6f}",
                "score_count": 0, "score_min": "", "score_max": "", "unique_scores": 0,
                "edges": 0, "hard_edges": 0, "soft_edges": 0,
                "error": "cannot parse N",
                "output_json": str(out_path),
            })
            print(f"[FAILSAFE] ({idx}/{len(files)}) {instance_name} -> cannot parse N", file=sys.stderr)
            continue

        # per-instance structured output schema to force all score keys
        format_schema = make_instance_format_schema(base_schema, n, force_edges_empty=True)

        schema_in_prompt = json.dumps(base_schema, ensure_ascii=False, indent=2) if args.inject_schema_in_prompt else ""

        mapping = {
            "{{LLM_ID}}": llm_id,
            "{{INSTANCE_NAME}}": instance_name,
            "{{TIMESTAMP}}": (timestamp or ""),
            "{{SCHEMA_JSON}}": schema_in_prompt,
            "{{INSTANCE_TEXT}}": sm_text,
        }

        system_prompt = render_template(system_tmpl, mapping) if system_tmpl else ""
        user_prompt = render_template(user_tmpl, mapping)

        # Make it unambiguous and short (helps local models)
        user_prompt += f"\n\nCaller note: N={n}. Output scores for ALL keys \"2\"..\"{n-1}\". Set edges=[] (edges will be computed by caller)."
        if not args.inject_schema_in_prompt:
            user_prompt += "\nCaller note: The JSON schema is enforced externally. Output JSON only."

        base_messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            base_messages.append({"role": "system", "content": system_prompt})
        base_messages.append({"role": "user", "content": user_prompt})

        last_err: Optional[Exception] = None
        ok = False
        attempts_used = 0
        final_status = ""
        final_fallback = False
        final_scores: Dict[str, Any] = {}
        final_edges: List[Dict[str, Any]] = []
        final_error = ""

        for attempt in range(1, args.max_attempts + 1):
            attempts_used = attempt
            try:
                messages = list(base_messages)
                if attempt > 1:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Retry. Previous output failed checks.\n"
                            f"Error: {str(last_err)}\n"
                            "Output corrected JSON ONLY. Include ALL score keys."
                        )
                    })

                content = ollama_chat(
                    base_url=args.ollama_url,
                    model=args.model,
                    messages=messages,
                    format_schema=format_schema,
                    temperature=args.temperature,
                    seed=args.seed,
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                    timeout_sec=args.timeout_sec,
                )

                # raw save
                (raw_dir / f"{sm_path.stem}.attempt{attempt}.txt").write_text(content, encoding="utf-8")

                obj = parse_json_object(content)
                obj = canonicalize_meta(obj, instance_name, llm_id, timestamp)

                # score completeness / non-trivial
                scores = obj.get("scores") or {}
                passed, reason = check_scores_complete_and_nontrivial(scores, n)
                if not passed:
                    raise ValueError(reason)

                # overwrite edges deterministically
                obj["edges"] = build_edges_from_scores(sm_text, {k: float(v) for k, v in scores.items()}, args.max_edges_total, args.max_hard_edges, args.hard_delta_min, args.soft_delta_min)

                # validate against ORIGINAL schema.json
                errors = sorted(base_validator.iter_errors(obj), key=lambda e: list(e.path))
                if errors:
                    raise ValueError(f"Base schema validation failed: {errors[0].message}")

                out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                final_status = "ok"
                final_fallback = False
                final_scores = obj.get("scores") or {}
                final_edges = obj.get("edges") or []
                final_error = ""
                print(f"[OK] ({idx}/{len(files)}) {instance_name} -> {out_path.name} (edges={len(obj['edges'])})")
                ok = True
                last_err = None
                break

            except Exception as e:
                last_err = e
                print(f"[WARN] ({idx}/{len(files)}) {instance_name} attempt {attempt}/{args.max_attempts} failed: {e}",
                      file=sys.stderr)
                time.sleep(0.2)

        if not ok:
            if args.fallback_det_scores:
                det_scores = compute_strict_v3_scores(sm_text)
                if det_scores is not None:
                    obj = {
                        "instance": instance_name,
                        "scores": det_scores,
                        "edges": build_edges_from_scores(sm_text, det_scores, args.max_edges_total, args.max_hard_edges, args.hard_delta_min, args.soft_delta_min),
                        "meta": {"llm": llm_id + "+fallback_det_scores", "prompt_version": "rcpsp-mixed-strict-v3"}
                    }
                    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                    failures.append(instance_name + " (LLM failed -> deterministic strict-v3 scores)")
                    final_status = "fallback_det_scores"
                    final_fallback = True
                    final_scores = obj.get("scores") or {}
                    final_edges = obj.get("edges") or []
                    final_error = str(last_err) if last_err else "LLM failed"
                    print(f"[FALLBACK_DET_SCORES] ({idx}/{len(files)}) {instance_name} -> wrote deterministic scores (edges={len(obj['edges'])})",
                          file=sys.stderr)
                else:
                    fs = make_failsafe_json(instance_name, llm_id, n)
                    out_path.write_text(json.dumps(fs, ensure_ascii=False, indent=2), encoding="utf-8")
                    failures.append(instance_name + " (LLM failed + deterministic parse failed -> FAIL-SAFE)")
                    final_status = "failsafe"
                    final_fallback = True
                    final_scores = fs.get("scores") or {}
                    final_edges = fs.get("edges") or []
                    final_error = str(last_err) if last_err else "LLM failed and deterministic parse failed"
                    print(f"[FAILSAFE] ({idx}/{len(files)}) {instance_name} -> wrote FAIL-SAFE",
                          file=sys.stderr)
            else:
                fs = make_failsafe_json(instance_name, llm_id, n)
                out_path.write_text(json.dumps(fs, ensure_ascii=False, indent=2), encoding="utf-8")
                failures.append(instance_name)
                final_status = "failsafe"
                final_fallback = True
                final_scores = fs.get("scores") or {}
                final_edges = fs.get("edges") or []
                final_error = str(last_err) if last_err else "LLM failed"
                print(f"[FAILSAFE] ({idx}/{len(files)}) {instance_name} -> wrote FAIL-SAFE",
                      file=sys.stderr)

        sc_n, sc_min, sc_max, sc_unique = score_stats(final_scores)
        e_total, e_hard, e_soft = edge_stats(final_edges)
        write_runtime_row({
            "dataset": args.dataset or "",
            "instance": instance_name,
            "status": final_status or ("ok" if ok else "unknown"),
            "fallback": final_fallback,
            "attempts_used": attempts_used,
            "elapsed_sec": f"{time.perf_counter() - inst_t0:.6f}",
            "score_count": sc_n, "score_min": sc_min, "score_max": sc_max, "unique_scores": sc_unique,
            "edges": e_total, "hard_edges": e_hard, "soft_edges": e_soft,
            "error": final_error,
            "output_json": str(out_path),
        })

        if args.sleep > 0:
            time.sleep(args.sleep)

    if failures:
        (out_dir / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"[INFO] failures written: {out_dir / 'failures.txt'} ({len(failures)} lines)")
    else:
        print("[INFO] no failures")


if __name__ == "__main__":
    main()
