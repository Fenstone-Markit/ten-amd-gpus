#!/usr/bin/env python3
# module-timing-02-edit.py <patch dir>: on top of claude-mt-01 [claude-mt-02]: GLM5_MT_MIN_TOKENS (default 65: prompt steps only)
# sets the smallest step that is timed, so decode steps can be timed too when CUDA graphs are off (graph replays run no Python);
# GLM5_MT_PER_LAYER=1 keeps the layer index in each key (e.g. FP8 layers' experts apart from the 4-bit ones).
import sys
p = sys.argv[1] + "/vllm/v1/worker/gpu_worker.py"; s = open(p).read()
a = "mt = _GLM5_MODULE_TIMING and scheduler_output.total_num_scheduled_tokens > 64  # [claude-mt-01]"
assert s.count(a) == 1
s = s.replace(a, "mt = _GLM5_MODULE_TIMING and scheduler_output.total_num_scheduled_tokens >= _GLM5_MT_MIN_TOKENS  # [claude-mt-01/02]")
b = "                key = f\"{type(mod).__name__}:{m.group(2) or '(whole layer)'}\"\n"
assert s.count(b) == 1
s = s.replace(b, "                key = f\"{type(mod).__name__}:{m.group(2) or '(whole layer)'}\"\n                if _GLM5_MT_PER_LAYER:  # [claude-mt-02]\n                    key = f\"L{int(m.group(1)):02d} \" + key\n")
c = '_GLM5_MODULE_TIMING = os.environ.get("GLM5_MODULE_TIMING") == "1"  # [claude-mt-01]\n'
assert s.count(c) == 1
s = s.replace(c, c + '_GLM5_MT_MIN_TOKENS = int(os.environ.get("GLM5_MT_MIN_TOKENS", "65") or 65)  # [claude-mt-02]\n_GLM5_MT_PER_LAYER = os.environ.get("GLM5_MT_PER_LAYER") == "1"  # [claude-mt-02]\n')
open(p, "w").write(s); print("edited", p)
