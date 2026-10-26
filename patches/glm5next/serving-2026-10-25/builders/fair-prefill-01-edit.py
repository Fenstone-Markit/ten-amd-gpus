#!/usr/bin/env python3
# fair-prefill-01-edit.py <patch dir>: v1/engine/core.py [claude-fair-01]. GLM5_PREFILL_EVERY_MS=<ms> (0 or unset = off,
# vLLM's behaviour exactly). While one request is generating and another's prompt is being read, a step that carries a prompt
# chunk is allowed at most once every <ms> milliseconds; the steps in between are decode-only (CUDA graph, about 46 ms each on
# 3x4), so the generating request keeps a usable speed. Uses vLLM's own prefill-deferral path (schedule(throttle_prefills=True):
# in-progress prompt chunks and new admissions wait, decodes run; with no request generating nothing is deferred), which the
# base EngineCore never switches on (its _should_throttle_prefills returns False; only the data-parallel engine overrides it).
# Measured before the change (glm-serve-44): a mixed step takes about 1.8 s and a generating request got about 1 token per step.
import sys
p = sys.argv[1] + "/vllm/v1/engine/core.py"; s = open(p).read()
a = '''    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        return False
'''
assert s.count(a) == 1, "anchor"
s = s.replace(a, '''    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise.
        [claude-fair-01] GLM5_PREFILL_EVERY_MS > 0: release prompt chunks at most once per that many ms
        while a request is decoding (the scheduler defers prefills only when one is)."""
        if _GLM5_PREFILL_EVERY_MS <= 0:
            return False
        now = time.monotonic()
        if now - getattr(self, "_glm5_last_release", 0.0) >= _GLM5_PREFILL_EVERY_MS / 1000.0:
            self._glm5_last_release = now
            return False
        return True
''')
m = '_GLM5_STEP_TIMING = os.environ.get("GLM5_STEP_TIMING") == "1"  # [claude-mtp-06]\n'
assert s.count(m) == 1
s = s.replace(m, m + '_GLM5_PREFILL_EVERY_MS = float(os.environ.get("GLM5_PREFILL_EVERY_MS", "0") or 0)  # [claude-fair-01]\n')
open(p, "w").write(s); print("edited", p)
