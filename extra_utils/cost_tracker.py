"""
cost_tracker.py
────────────────────────────────────────────────────────────────────────────
Process-independent LiteLLM cost & usage tracker.

Usage:
    from cost_tracker import init_tracker, get_tracker

    init_tracker(log_dir="./logs")          # call once at startup
    # ... all litellm / DSPy calls happen ...
    print(get_tracker().summary())
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import litellm


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    ts:                 str           # timestamp_utc
    tags:               dict          # caller-supplied name/metadata — used as log filename too
    model:              str
    prompt_tokens:      int
    completion_tokens:  int
    latency_sec:        float
    cost_usd:           float


# ── Tracker ──────────────────────────────────────────────────────────────────

class CostTracker:
    """
    Thread-safe, process-independent cost tracker.

    Writes two files per run:
      <log_dir>/<tag_name>.jsonl   — one line per call (lightweight, appendable)
      <log_dir>/<tag_name>_summary.json — running totals only (tiny)

    The filename is derived from the 'name' tag (or 'default' if not set).
    Multiple processes writing to the same name all append to the same .jsonl
    safely — append is atomic on all major OS/filesystems for small writes.
    """

    _instance:  Optional["CostTracker"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __new__(cls, log_dir: str = "./llm_logs"):
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._ready = False
                cls._instance = inst
        return cls._instance

    def _setup(self, log_dir: str, tags: dict):
        if self._ready:
            return
        self._lock    = threading.Lock()
        self._tags    = tags
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Derive filename from 'name' tag — fallback to 'default'
        self._name         = tags.get("name", "default")
        self._calls_path   = self._log_dir / f"{self._name}.jsonl"
        self._summary_path = self._log_dir / f"{self._name}_summary.json"

        # In-memory running totals (rebuilt from summary on load)
        self.total_cost_usd: float = 0.0
        self.total_calls:    int   = 0

        self._load_summary()
        self._register_litellm()
        self._ready = True
        print(f"📊 CostTracker [{self._name}] | calls→ {self._calls_path}")

    # ── persistence ─────────────────────────────────────────────────────────

    def _load_summary(self):
        """Restore running totals from summary file on startup."""
        if self._summary_path.exists():
            try:
                data = json.loads(self._summary_path.read_text())
                self.total_cost_usd = data.get("total_cost_usd", 0.0)
                self.total_calls    = data.get("total_calls",    0)
                # print(f"   ↳ Resumed: {self.total_calls} calls, ${self.total_cost_usd:.5f}")
            except Exception as e:
                print(f"⚠️  Could not load summary ({e}) — starting fresh")

    def _append_call(self, record: CallRecord):
        """Append one JSON line to the .jsonl file — atomic enough for multi-process."""
        line = json.dumps(asdict(record), separators=(",", ":")) + "\n"
        with open(self._calls_path, "a") as f:
            f.write(line)

    def _save_summary(self):
        """Overwrite summary with current totals (called under lock)."""
        self._summary_path.write_text(json.dumps({
            "name":           self._name,
            "tags":           self._tags,
            "total_calls":    self.total_calls,
            "total_cost_usd": round(self.total_cost_usd, 8),
        }, indent=2))

    # ── record ───────────────────────────────────────────────────────────────

    def add(self, record: CallRecord):
        with self._lock:
            self.total_cost_usd += record.cost_usd
            self.total_calls    += 1
            self._append_call(record)
            self._save_summary()

        # print(
        #     f"💰 ${record.cost_usd:.6f}"
        #     f" | in:{record.prompt_tokens} out:{record.completion_tokens} tok"
        #     f" | {record.latency_sec:.2f}s [{record.model}]"
        #     f" | Σ ${self.total_cost_usd:.5f} ({self.total_calls} calls)"
        # )

    def summary(self) -> dict:
        with self._lock:
            return {
                "name":           self._name,
                "tags":           self._tags,
                "total_calls":    self.total_calls,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "calls_log":      str(self._calls_path),
                "summary_log":    str(self._summary_path),
            }

    def set_tags(self, **kwargs):
        """Update tags attached to subsequent call records."""
        self._tags.update(kwargs)

    # ── litellm wiring ───────────────────────────────────────────────────────

    def _register_litellm(self):
        tracker = self

        def on_success(kwargs, response, start_time, end_time):
            try:
                cost = float(
                    response._hidden_params.get("response_cost") or
                    litellm.completion_cost(completion_response=response) or
                    0.0
                )
                usage             = getattr(response, "usage", None)
                prompt_tokens     = getattr(usage, "prompt_tokens",     0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                model = (getattr(response, "model", "") or kwargs.get("model", "unknown")).split("/")[-1]

                tracker.add(CallRecord(
                    ts                = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    tags              = dict(tracker._tags),
                    model             = model,
                    prompt_tokens     = prompt_tokens,
                    completion_tokens = completion_tokens,
                    latency_sec       = round((end_time - start_time).total_seconds(), 3),
                    cost_usd          = round(cost, 8),
                ))
            except Exception as e:
                print(f"⚠️  CostTracker on_success error: {e}")

        def on_failure(kwargs, exception, start_time, end_time):
            elapsed = round((end_time - start_time).total_seconds(), 2)
            print(f"❌ [{kwargs.get('model','?')}] failed after {elapsed}s — {type(exception).__name__}: {exception}")

        litellm.success_callback = [on_success]
        litellm.failure_callback = [on_failure]


# ── Public API ───────────────────────────────────────────────────────────────

def init_tracker(log_dir: str = "./llm_logs", **default_tags) -> CostTracker:
    """
    Initialize (or return existing) tracker for this process.

    Args:
        log_dir:      Directory where JSON logs are written.
        **default_tags: Arbitrary key=value metadata attached to every call.
                        e.g. init_tracker(log_dir="./logs", pipeline="extraction", version="v2")

    Returns:
        CostTracker singleton for this process.
    """
    tracker = CostTracker(log_dir=log_dir)
    tracker._setup(log_dir=log_dir, tags=default_tags)
    return tracker


def get_tracker() -> CostTracker:
    """Return the already-initialized tracker. Raises if init_tracker() not called first."""
    if CostTracker._instance is None or not CostTracker._instance._ready:
        raise RuntimeError("CostTracker not initialized. Call init_tracker(log_dir=...) first.")
    return CostTracker._instance