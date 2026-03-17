#!/usr/bin/env python3
"""
Estimate per-file and aggregate token usage + USD cost for gpt-4.1 across
super_bench/runs/progress/**/*.json logs which follow the optimized_react schema.

Priority path uses LiteLLM (if installed) for tokenization + pricing.
Fallback path uses a simple heuristic tokenizer and configurable per-token pricing.

Assumptions for mapping trajectory -> model turns:
- Each `thought_i` is treated as an assistant completion for turn i.
- Prompt for i=0 is `example.query`.
- Prompt for i>0 is prior tool context: a compact string built from
  `tool_name_{i-1}`, `tool_args_{i-1}` (json), and `observation_{i-1}`.

Outputs a text report at repo root: cost_estimates_gpt4.1.txt
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]  # repo root (gepa-modified)
PROGRESS_DIR = ROOT / "super_bench" / "runs" / "progress"
REPORT_PATH = ROOT / "cost_estimates_gpt4.1.txt"


# ------------------------- Pricing helpers -------------------------

DEFAULT_INPUT_PER_TOKEN = float(os.getenv("GPT41_INPUT_COST_PER_TOKEN", "0.000005"))
DEFAULT_OUTPUT_PER_TOKEN = float(os.getenv("GPT41_OUTPUT_COST_PER_TOKEN", "0.000015"))


@dataclass
class Pricing:
    input_cost_per_token: float
    output_cost_per_token: float
    source: str


def resolve_pricing() -> Pricing:
    """Pick pricing from litellm if available, else from env/defaults."""
    try:
        import litellm  # type: ignore
        # Ensure model is registered; if community map lacks gpt-4.1, fall back to env/defaults
        mc: Dict[str, Any] = getattr(litellm, "model_cost", {}) or {}
        if "gpt-4.1" in mc:
            m = mc["gpt-4.1"]
            return Pricing(
                input_cost_per_token=float(m["input_cost_per_token"]),
                output_cost_per_token=float(m["output_cost_per_token"]),
                source="litellm.model_cost",
            )
        # Fallback to env/defaults
        return Pricing(DEFAULT_INPUT_PER_TOKEN, DEFAULT_OUTPUT_PER_TOKEN, "env/default")
    except Exception:
        return Pricing(DEFAULT_INPUT_PER_TOKEN, DEFAULT_OUTPUT_PER_TOKEN, "env/default")


# ------------------------- Token counting -------------------------

def count_tokens_litellm(prompt: str, completion: str) -> Tuple[int, int, str]:
    try:
        import litellm  # type: ignore
        # token_counter counts across messages; we separate prompt and completion
        ptoks = litellm.token_counter(model="gpt-4.1", messages=[{"role": "user", "content": prompt}])
        ctoks = litellm.token_counter(model="gpt-4.1", messages=[{"role": "assistant", "content": completion}])
        return int(ptoks or 0), int(ctoks or 0), "litellm"
    except Exception:
        return count_tokens_fallback(prompt, completion) + ("fallback",)


def rough_token_len(text: str) -> int:
    # Heuristic ~4 chars per token, clamp non-empty lines
    if not text:
        return 0
    n = max(1, int(len(text) / 4))
    return n


def count_tokens_fallback(prompt: str, completion: str) -> Tuple[int, int]:
    return rough_token_len(prompt), rough_token_len(completion)


# ------------------------- Parsing helpers -------------------------

def compact_observation(prev_tool: Optional[str], prev_args: Any, obs: Any) -> str:
    parts = []
    if prev_tool:
        parts.append(f"tool={prev_tool}")
    if prev_args is not None:
        try:
            parts.append("args=" + json.dumps(prev_args, ensure_ascii=False)[:4000])
        except Exception:
            parts.append(f"args={str(prev_args)[:4000]}")
    if obs is not None:
        if isinstance(obs, dict):
            payload = obs.get("output") or obs.get("content") or obs.get("message") or obs.get("error")
            if payload is None:
                payload = obs
            try:
                parts.append("obs=" + (payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))[:8000])
            except Exception:
                parts.append("obs=" + str(payload)[:8000])
        else:
            parts.append("obs=" + str(obs)[:8000])
    return "\n".join(parts)


def iter_thought_indices(traj: Dict[str, Any]):
    for k in sorted(traj.keys()):
        if k.startswith("thought_"):
            try:
                yield int(k.split("_")[1])
            except Exception:
                continue


def extract_turn_context(item: Dict[str, Any], idx: int) -> Tuple[str, str]:
    traj = item.get("prediction", {}).get("trajectory", {})
    thought = traj.get(f"thought_{idx}") or ""
    if idx == 0:
        prompt = item.get("example", {}).get("query", "")
    else:
        prev_tool = traj.get(f"tool_name_{idx-1}")
        prev_args = traj.get(f"tool_args_{idx-1}")
        prev_obs = traj.get(f"observation_{idx-1}")
        prompt = compact_observation(prev_tool, prev_args, prev_obs)
    return prompt, thought


# ------------------------- Main estimation -------------------------

def estimate_for_json(path: Path, pricing: Pricing) -> Dict[str, Any]:
    with path.open() as f:
        try:
            data = json.load(f)
        except Exception as e:
            return {"file": str(path), "error": f"json_load_failed: {e}"}

    total_prompt_toks = 0
    total_completion_toks = 0
    total_cost = 0.0
    used_counter = None

    for item in data if isinstance(data, list) else [data]:
        traj = item.get("prediction", {}).get("trajectory", {})
        for i in iter_thought_indices(traj):
            prompt, completion = extract_turn_context(item, i)
            pt, ct, mode = count_tokens_litellm(prompt, completion)
            used_counter = used_counter or mode
            total_prompt_toks += pt
            total_completion_toks += ct
            total_cost += pt * pricing.input_cost_per_token + ct * pricing.output_cost_per_token

    return {
        "file": str(path),
        "prompt_tokens": total_prompt_toks,
        "completion_tokens": total_completion_toks,
        "total_cost_usd": total_cost,
        "token_counter": used_counter or "fallback",
    }


def main() -> int:
    pricing = resolve_pricing()
    json_files = sorted(PROGRESS_DIR.rglob("*.json"))
    results = []
    agg_prompt = agg_completion = 0
    agg_cost = 0.0

    for p in json_files:
        res = estimate_for_json(p, pricing)
        results.append(res)
        if "error" not in res:
            agg_prompt += res["prompt_tokens"]
            agg_completion += res["completion_tokens"]
            agg_cost += res["total_cost_usd"]

    lines = []
    lines.append("gpt-4.1 cost estimates via LiteLLM if available")
    lines.append(f"pricing_source: {pricing.source}")
    lines.append(f"input_cost_per_token: {pricing.input_cost_per_token}")
    lines.append(f"output_cost_per_token: {pricing.output_cost_per_token}")
    lines.append("")
    for r in results:
        if "error" in r:
            lines.append(f"{r['file']}: ERROR {r['error']}")
        else:
            lines.append(
                f"{r['file']} | prompt_toks={r['prompt_tokens']} | completion_toks={r['completion_tokens']} | cost=${r['total_cost_usd']:.6f} | counter={r['token_counter']}"
            )
    lines.append("")
    lines.append("TOTALS:")
    lines.append(f"prompt_toks={agg_prompt}")
    lines.append(f"completion_toks={agg_completion}")
    lines.append(f"cost=${agg_cost:.6f}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
