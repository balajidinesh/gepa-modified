

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

    def __new__(cls, max_requests_per_min=1000, max_tokens_per_min=3_000_000):
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




# from aicodetools import ClientManager 

# code_tool_manager = ClientManager(
#                 "super-bench:latest", base_log_dir="runs/super/"
#             )

# code_tool_client = code_tool_manager.get_client('initial')
import os
# os.environ['OPENAI_API_KEY'/] = input()
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import dspy

lm = dspy.LM("azure/gpt-4.1", temperature=1.0, num_retries=30, callbacks=[DelayAndLogCallback()])
tlm = dspy.LM("azure/gpt-4.1",temperature=1.0)
dspy.configure(lm=lm)



# print(lm("Say this is a test!") ) # => ['This is a test!']
print(lm(messages=[{"role": "user", "content": "Say this is a test!"}]))  # => ['This is a test!']
print(tlm(messages=[{"role": "user", "content": "t : Say this is a test!"}]))  # => ['This is a test!']
## Load the benchmark and view one example from the benchmark
