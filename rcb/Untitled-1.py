# %%
# %pip install python-dotenv
# %uv add dspy

# %%
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))


# %%
import time
import threading
import tiktoken
from collections import deque
import dspy
from dspy.utils.callback import BaseCallback


class SlidingWindowLimiter:
    """Rate limiter that enforces both request and token limits per rolling minute."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_requests_per_min=1000, max_tokens_per_min=2_000_000):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.max_requests = max_requests_per_min
            cls._instance.max_tokens = max_tokens_per_min
            cls._instance.requests = deque()  # [(timestamp, tokens)]
            cls._instance._lock = threading.Lock()
            cls._instance.encoder = tiktoken.get_encoding("cl100k_base")
        return cls._instance

    def _cleanup(self, now):
        """Remove entries older than 60s."""
        while self.requests and now - self.requests[0][0] > 60:
            self.requests.popleft()

    def _count(self):
        """Total requests & tokens in current 60s window."""
        total_tokens = sum(t for _, t in self.requests)
        return len(self.requests), total_tokens

    def acquire(self, tokens_used=0):
        """Wait until request fits in sliding 60s window."""
        with self._lock:
            while True:
                now = time.time()
                self._cleanup(now)
                req_count, token_count = self._count()

                # Can fit in current 60s window?
                if (req_count < self.max_requests and
                        token_count + tokens_used <= self.max_tokens):
                    # Record the new request
                    self.requests.append((now, tokens_used))
                    break  # proceed

                # Otherwise, figure out when we can retry
                oldest_time = self.requests[0][0]
                sleep_time = max(0.01, 60 - (now - oldest_time))
                print(f"⚠️ Throttling: sleeping {sleep_time:.2f}s (req={req_count}, tokens={token_count})")
                time.sleep(sleep_time)


class DelayAndLogCallback(BaseCallback):
    """DSPy callback using sliding window limiter."""

    def __init__(self):
        self.limiter = SlidingWindowLimiter()

    def _estimate_tokens(self, messages=None, prompt=None):
        """Estimate token usage using tiktoken."""
        text = ""
        if prompt:
            text = str(prompt)
        elif messages:
            # concatenate all message contents
            text = " ".join(m.get("content", "") for m in messages)
        return len(self.limiter.encoder.encode(text))

    def on_lm_start(self, *args, **kwargs):
        inputs = kwargs.get("inputs") or {}
        prompt = inputs.get("prompt")
        messages = inputs.get("messages")
        tokens_used = self._estimate_tokens(messages=messages, prompt=prompt)
        self.limiter.acquire(tokens_used=tokens_used)

    def on_lm_end(self, *args, **kwargs):
        pass


# %%
import os
# os.environ['OPENAI_API_KEY'/] = input()
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# %%
import dspy
lm = dspy.LM("azure/gpt-4.1", temperature=1.0, num_retries=30, callbacks=[DelayAndLogCallback()])
tlm = dspy.LM("azure/gpt-4.1",temperature=1.0)
dspy.configure(lm=lm)

# %%
# print(lm("Say this is a test!") ) # => ['This is a test!']
print(lm(messages=[{"role": "user", "content": "Say this is a test!"}]))  # => ['This is a test!']
print(tlm(messages=[{"role": "user", "content": "t : Say this is a test!"}]))  # => ['This is a test!']

# %% [markdown]
# ## Load the benchmark and view one example from the benchmark

# %%
from gepa_artifact.benchmarks.researchcodebench import benchmark as rc_metas

# %%
bench = rc_metas[0].benchmark()

# %%
len(bench.train_set), len(bench.val_set), len(bench.test_set)

# %%
import pprint
pprint.pprint(bench.train_set[0])

# %% [markdown]
# ## Load the program and display the program
# The program is a 3-module system, each of which handles the urgency, sentiment and categories classification respectively

# %%
program = rc_metas[0].program[0]
program

# %% [markdown]
# ### Make Sure docker is installed and running

# %% [markdown]
# ## Define an evaluator and evaluate the base program

# %%
import dspy
evaluate = dspy.Evaluate(
    devset=bench.test_set,
    metric=rc_metas[0].metric,
    num_threads=4,
    display_table=True,
    display_progress=True,
    max_errors=100 * len(bench.test_set),
    provide_traceback = True,
    failure_score=0,
    save_as_json='rcb.json',
    save_as_csv='rcb.csv'
)

# %% [markdown]
# ## Load the GEPA Optimizer

# %%

import dspy
from gepa_artifact.gepa.gepa import GEPA,GEPAState
from gepa_artifact.utils.capture_stream_logger import Logger

import time

# %%


runs_dir = os.path.join(os.getcwd(), "runs", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(runs_dir, exist_ok=True)

gepa_logger = Logger(os.path.join(runs_dir, "run_log.txt"))

if rc_metas[0].feedback_fn_maps is None or rc_metas[0].feedback_fn_maps[0] is None:
    def feedback_func(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
        pred = rc_metas[0].metric_with_feedback(module_inputs, module_outputs, None)
        return {
            "feedback_score": pred.score,
            "feedback_text": pred.feedback,
        }

    feedback_fn_map = {k:feedback_func for k, v in program.named_predictors()}
else:
    feedback_fn_map = rc_metas[0].feedback_fn_maps[0]

optimizer = GEPA(
    named_predictor_to_feedback_fn_map=feedback_fn_map,
    knowledgebase_qe=None,
    metric=rc_metas[0].metric,
    run_linearized_gepa=False,
    use_merge=True, 
    teacher_lm = tlm,
    set_for_merge_minibatch='val', 
    track_scores_on='val',
    num_iters=20,
    run_dir=runs_dir,
    logger=gepa_logger,
    num_threads=4)

# %% [markdown]
# ## Optimize the program with GEPA

# %%
rc_metas[0].program[0].get_lm()

# %% [markdown]
# ## Load from the Saved dir

# %%
state = GEPAState.load('runs/rcb-w-gold')

# %%
state.total_num_evals

# %%
def idxmax(lst):
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)

# %%
gepa_state = state
best_prog_idx = idxmax(gepa_state.per_program_tracked_scores)
best_progs = gepa_state.program_candidates
best_prog = best_progs[best_prog_idx]

# %%
len(best_progs)

# %%
# # best_progs
# for idx,i in enumerate(best_progs):
#     print(f'*************{idx}***************')
#     for name, pred in i.named_predictors():
#         print("================================")
#         print(f"Predictor: {name}")
#         print("================================")
#         print("Prompt:")
#         print(pred.signature.instructions)
#         print("*********************************")

# %%
optimized_program=best_prog

# %%
best_prog_idx

# %% [markdown]
# ### Let's print the prompts that GEPA discovered

# %%
for name, pred in optimized_program.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")

# %% [markdown]
# ## Now, let's evaluate the optimized program

# %%


# %%
import dspy
evaluate = dspy.Evaluate(
    devset=bench.test_set,
    metric=rc_metas[0].metric,
    num_threads=4,
    display_table=True,
    display_progress=True,
    max_errors=100 * len(bench.test_set),
    provide_traceback = True,
    failure_score=0,
    save_as_json='optimized_rcb.json',
    save_as_csv='optimized_rcb.csv'
)

# %%
# latest_program = best_progs[-1]

# for name, pred in latest_program.named_predictors():
#     print("================================")
#     print(f"Predictor: {name}")
#     print("================================")
#     print("Prompt:")
#     print(pred.signature.instructions)
#     print("*********************************")

# %% [markdown]
# ## if we use the best program given by gepa

# %%
evaluate(optimized_program)

# %%
latest_program = best_progs[-1]

for name, pred in latest_program.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")

# %%
# evaluate(latest_program)

# %%


# %% [markdown]
# 


