"""
usage_examples.py
────────────────────────────────────────────────────────────────────────────
How to use cost_tracker.py — from minimal to multi-process.
────────────────────────────────────────────────────────────────────────────
"""
import extra_util.cost_tracker as cost_tracker
import sys

sys.modules["cost_tracker"] = cost_tracker
# ============================================================================
# EXAMPLE 1 — Minimal
# Just give it a name and a dir. Done.
# ============================================================================

from cost_tracker import init_tracker, get_tracker
import dspy

init_tracker(log_dir="./logs", name="my_pipeline")

lm = dspy.LM("azure/gpt-4.1", temperature=1.0, max_tokens=8192)
dspy.configure(lm=lm)

predictor = dspy.Predict("question -> answer")
predictor(question="What is the capital of France?")

print(get_tracker().summary())

# Console output per call:
#   💰 $0.000214 | in:48 out:12 tok | 1.32s [gpt-4.1] | Σ $0.00021 (1 calls)

# summary() output:
#   {
#     "name": "my_pipeline",
#     "tags": {"name": "my_pipeline"},
#     "total_calls": 1,
#     "total_cost_usd": 0.000214,
#     "calls_log": "logs/my_pipeline.jsonl",
#     "summary_log": "logs/my_pipeline_summary.json"
#   }

# Files written:
#   logs/my_pipeline.jsonl          ← one line per call
#   logs/my_pipeline_summary.json   ← running totals only


# ============================================================================
# EXAMPLE 2 — Extra tags for filtering/analysis later
# ============================================================================

from cost_tracker import init_tracker
import dspy

init_tracker(
    log_dir="./logs",
    name="doc_extraction",
    version="v2.1",
    env="prod",
)

lm = dspy.LM("azure/gpt-4.1", max_tokens=8192)
dspy.configure(lm=lm)

dspy.Predict("document -> entities")(document="Alice went to Paris in 2024.")

# Each line in doc_extraction.jsonl looks like:
# {"ts":"2026-03-07T10:22:01Z","tags":{"name":"doc_extraction","version":"v2.1","env":"prod"},
#  "model":"gpt-4.1","prompt_tokens":55,"completion_tokens":18,"latency_sec":1.21,"cost_usd":0.00025400}


# ============================================================================
# EXAMPLE 3 — Change tags mid-run per stage
# ============================================================================

from cost_tracker import init_tracker, get_tracker
import dspy

tracker = init_tracker(log_dir="./logs", name="multi_stage", version="v1")
lm = dspy.LM("azure/gpt-4.1", max_tokens=8192)
dspy.configure(lm=lm)

predict = dspy.Predict("text -> result")

tracker.set_tags(stage="extraction")
predict(text="Extract: Alice went to Paris.")

tracker.set_tags(stage="summarization")
predict(text="Summarize: The quick brown fox jumps.")

tracker.set_tags(stage="validation")
predict(text="Validate: Is '10 Downing St' a valid address?")

# All 3 calls land in logs/multi_stage.jsonl
# Each line carries its own stage tag — easy to grep/filter later:
#   grep '"stage":"extraction"' logs/multi_stage.jsonl


# ============================================================================
# EXAMPLE 4 — Multiple parallel processes (multiprocessing.Pool)
#
# .jsonl append is atomic for small writes on Linux/macOS/Windows.
# All processes safely write to the same file — no coordination needed.
# ============================================================================

import multiprocessing
from cost_tracker import init_tracker
import dspy

def worker(doc: str, worker_id: int):
    # Each worker calls init_tracker independently — singleton is per-process.
    init_tracker(log_dir="./logs/parallel", name="parallel_run", worker=str(worker_id))

    lm = dspy.LM("azure/gpt-4.1", max_tokens=8192)
    dspy.configure(lm=lm)

    result = dspy.Predict("doc -> entities")(doc=doc)
    return result.entities


if __name__ == "__main__":
    docs = ["Doc about Paris.", "Doc about Tokyo.", "Doc about Cairo."]
    with multiprocessing.Pool(3) as pool:
        results = pool.starmap(worker, [(d, i) for i, d in enumerate(docs)])

    # After all workers finish:
    #   logs/parallel/parallel_run.jsonl         ← all 3 workers' calls, merged
    #   logs/parallel/parallel_run_summary.json  ← last writer's running totals


# ============================================================================
# EXAMPLE 5 — Full setup: rate limiter + cost tracker together
# ============================================================================

from cost_tracker import init_tracker, get_tracker
from collections import deque
import threading, time, tiktoken
import dspy
from dspy.utils.callback import BaseCallback


class SlidingWindowLimiter:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_requests_per_min=1000, max_tokens_per_min=2_000_000):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.max_requests = max_requests_per_min
            cls._instance.max_tokens = max_tokens_per_min
            cls._instance.requests = deque()
            cls._instance._lock = threading.Lock()
            cls._instance.encoder = tiktoken.get_encoding("cl100k_base")
        return cls._instance

    def _cleanup(self, now):
        while self.requests and now - self.requests[0][0] > 60:
            self.requests.popleft()

    def acquire(self, tokens_used=0):
        with self._lock:
            while True:
                now = time.time()
                self._cleanup(now)
                req_count   = len(self.requests)
                token_count = sum(t for _, t in self.requests)
                if req_count < self.max_requests and token_count + tokens_used <= self.max_tokens:
                    self.requests.append((now, tokens_used))
                    break
                sleep_time = max(0.01, 60 - (now - self.requests[0][0]))
                print(f"⚠️ Throttling: {sleep_time:.2f}s")
                time.sleep(sleep_time)


class RateLimitCallback(BaseCallback):
    """DSPy callback for rate limiting only. Cost tracking handled by CostTracker."""

    def __init__(self):
        self.limiter = SlidingWindowLimiter()

    def on_lm_start(self, *args, **kwargs):
        inputs = kwargs.get("inputs") or {}
        msgs   = inputs.get("messages") or []
        text   = inputs.get("prompt") or " ".join(m.get("content", "") for m in msgs)
        tokens = len(self.limiter.encoder.encode(str(text)))
        self.limiter.acquire(tokens_used=tokens)

    def on_lm_end(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    import json

    tracker = init_tracker(log_dir="./logs", name="full_example", pipeline="qa", version="v3")

    lm = dspy.LM("azure/gpt-4.1", temperature=1.0, max_tokens=8192, callbacks=[RateLimitCallback()])
    dspy.configure(lm=lm)

    predictor = dspy.Predict("question -> answer")
    for q in ["What is ML?", "What is an LLM?", "What is DSPy?"]:
        predictor(question=q)

    print("\n── Summary ──")
    print(json.dumps(tracker.summary(), indent=2))