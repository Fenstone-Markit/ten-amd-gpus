#!/usr/bin/env python3
# module-timing-01-edit.py <patch dir>: gpu_worker.py GLM5_MODULE_TIMING=1 (off unless set) [claude-mt-01]. Needs the
# claude-stt-01 wrapper (stage-timing-01-edit.py) already in the file.
# GPU time per model part during prompt steps (more than 64 tokens; decode steps replay CUDA graphs and run no Python):
# on the first such step every module named layers.<N>[.<a>[.<b>[.<c>]]] gets a forward pre-hook and post-hook that record
# CUDA events on the current stream (the model is not torch.compiled here, compilation mode NONE). After the step's GPU work is
# done (current stream synchronize) each pair gives that call's GPU time; totals are kept per <class>:<path below the layer
# index>, so all layers of one kind add up. Tensor-parallel rank 0 of each stage writes the table to
# /cache/module-timing/mt-pp<stage>.txt every 5 timed steps. Nested parts overlap (a layer's time includes its attention's).
import sys
p = sys.argv[1] + "/vllm/v1/worker/gpu_worker.py"; s = open(p).read()
a = '''    def execute_model(self, scheduler_output: "SchedulerOutput"):
        if not _GLM5_STAGE_TIMING:
            return self._glm5_execute_model_inner(scheduler_output)
        t0 = time.perf_counter()
        out = self._glm5_execute_model_inner(scheduler_output)
        torch.cuda.current_stream().synchronize()
        self._glm5_stt("exec", t0, scheduler_output)
        return out
'''
assert s.count(a) == 1, "needs claude-stt-01"
s = s.replace(a, '''    def execute_model(self, scheduler_output: "SchedulerOutput"):
        if not (_GLM5_STAGE_TIMING or _GLM5_MODULE_TIMING):
            return self._glm5_execute_model_inner(scheduler_output)
        mt = _GLM5_MODULE_TIMING and scheduler_output.total_num_scheduled_tokens > 64  # [claude-mt-01]
        if mt:
            self._glm5_mt_begin()
        t0 = time.perf_counter()
        out = self._glm5_execute_model_inner(scheduler_output)
        torch.cuda.current_stream().synchronize()
        if mt:
            self._glm5_mt_end(scheduler_output, (time.perf_counter() - t0) * 1000)
        if _GLM5_STAGE_TIMING:
            self._glm5_stt("exec", t0, scheduler_output)
        return out

    def _glm5_mt_begin(self):  # [claude-mt-01]
        if getattr(self, "_glm5_mt", None) is None:
            import re
            from vllm.distributed import get_tensor_model_parallel_rank
            st = self._glm5_mt = {"on": False, "pairs": [], "open": {}, "agg": {}, "steps": 0, "tok": 0, "wall": 0.0,
                                  "rank0": get_tensor_model_parallel_rank() == 0}
            pat = re.compile(r"(?:^|\\.)layers\\.(\\d+)(?:\\.(.*))?$")
            def pre(key):
                def f(mod, args):
                    if st["on"]:
                        e = torch.cuda.Event(enable_timing=True); e.record(); st["open"].setdefault(key, []).append(e)
                return f
            def post(key):
                def f(mod, args, output):
                    if st["on"] and st["open"].get(key):
                        e = torch.cuda.Event(enable_timing=True); e.record(); st["pairs"].append((key, st["open"][key].pop(), e))
                return f
            n = 0
            for name, mod in self.model_runner.model.named_modules():
                m = pat.search(name)
                if not m or (m.group(2) or "").count(".") > 2:
                    continue
                key = f"{type(mod).__name__}:{m.group(2) or '(whole layer)'}"
                mod.register_forward_pre_hook(pre(key)); mod.register_forward_hook(post(key)); n += 1
            logger.info("[claude-mt-01] module timing hooks on %d modules", n)
        st = self._glm5_mt
        st["on"] = True; st["pairs"] = []; st["open"] = {}

    def _glm5_mt_end(self, so, wall_ms):  # [claude-mt-01]
        st = self._glm5_mt; st["on"] = False
        for key, a, b in st["pairs"]:
            v = st["agg"].setdefault(key, [0.0, 0]); v[0] += a.elapsed_time(b); v[1] += 1
        st["pairs"] = []; st["steps"] += 1; st["tok"] += so.total_num_scheduled_tokens; st["wall"] += wall_ms
        if st["rank0"] and st["steps"] % 5 == 0:
            os.makedirs("/cache/module-timing", exist_ok=True)
            rows = sorted(st["agg"].items(), key=lambda kv: -kv[1][0])
            with open(f"/cache/module-timing/mt-pp{get_pp_group().rank_in_group}.txt", "w") as f:
                f.write(f"{st['steps']} prompt steps, {st['tok']} tokens, stage wall time {st['wall']:.0f} ms "
                        f"({st['wall'] / st['steps']:.1f} ms per step)\\n")
                f.write("| part | GPU ms total | calls | ms per call | % of stage wall time |\\n| --- | --- | --- | --- | --- |\\n")
                for key, (ms, c) in rows:
                    f.write(f"| {key} | {ms:.1f} | {c} | {ms / c:.3f} | {100 * ms / max(st['wall'], 1e-9):.1f} |\\n")
''')
m = '_GLM5_STAGE_TIMING = os.environ.get("GLM5_STAGE_TIMING") == "1"  # [claude-stt-01]\n'
assert s.count(m) == 1
s = s.replace(m, m + '_GLM5_MODULE_TIMING = os.environ.get("GLM5_MODULE_TIMING") == "1"  # [claude-mt-01]\n')
open(p, "w").write(s); print("edited", p)
