#!/usr/bin/env python3
# stage-timing-01-edit.py <patch dir>: gpu_worker.py gets GLM5_STAGE_TIMING=1 (off unless set) [claude-stt-01].
# Every worker of tensor-parallel rank 0 (one per pipeline stage) appends a line per call to /cache/stage-timing/st-pp<stage>.txt:
# the call (exec = run the stage's layers, samp = sampling, plus drafting on the last stage; on earlier stages samp is the wait for
# the last stage's spec results), its time in ms (waiting for this stage's own GPU work: current stream only, so the
# non-blocking send to the next stage is not waited for), the idle time since the previous call, and the step's tokens,
# decode requests (3 tokens or fewer) and prompt requests. For finding why a step that mixes a prompt chunk with another
# request's decode takes 3.9 s (alone a 1152-token chunk takes about 0.44 s per stage).
import sys
p = sys.argv[1] + "/vllm/v1/worker/gpu_worker.py"; s = open(p).read()
a = '    @torch.inference_mode()\n    @with_gpu_sync_check\n    def execute_model(\n        self, scheduler_output: "SchedulerOutput"\n    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:\n'
b = '    @torch.inference_mode()\n    @with_gpu_sync_check\n    def sample_tokens(\n        self, grammar_output: "GrammarOutput | None"\n    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:\n        return self.model_runner.sample_tokens(grammar_output)\n'
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, a.replace("def execute_model(", "def _glm5_execute_model_inner("))
s = s.replace(b, b.replace("def sample_tokens(", "def _glm5_sample_tokens_inner(") + '''
    # [claude-stt-01] GLM5_STAGE_TIMING=1: time every stage call (see stage-timing-01-edit.py). Off: a plain pass-through.
    def execute_model(self, scheduler_output: "SchedulerOutput"):
        if not _GLM5_STAGE_TIMING:
            return self._glm5_execute_model_inner(scheduler_output)
        t0 = time.perf_counter()
        out = self._glm5_execute_model_inner(scheduler_output)
        torch.cuda.current_stream().synchronize()
        self._glm5_stt("exec", t0, scheduler_output)
        return out

    def sample_tokens(self, grammar_output: "GrammarOutput | None"):
        if not _GLM5_STAGE_TIMING:
            return self._glm5_sample_tokens_inner(grammar_output)
        t0 = time.perf_counter()
        out = self._glm5_sample_tokens_inner(grammar_output)
        torch.cuda.current_stream().synchronize()
        self._glm5_stt("samp", t0, None)
        return out

    def _glm5_stt(self, kind, t0, so):
        t1 = time.perf_counter()
        f = getattr(self, "_glm5_stt_f", None)
        if f is None:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() != 0:
                self._glm5_stt_f = False
                return
            os.makedirs("/cache/stage-timing", exist_ok=True)
            f = self._glm5_stt_f = open(f"/cache/stage-timing/st-pp{get_pp_group().rank_in_group}.txt", "a", buffering=1)
            self._glm5_stt_prev = t0
        if f is False:
            return
        gap = (t0 - self._glm5_stt_prev) * 1000
        self._glm5_stt_prev = t1
        extra = ""
        if so is not None:
            d = so.num_scheduled_tokens
            nd = sum(1 for v in d.values() if v <= 3)
            extra = f" tok={so.total_num_scheduled_tokens} dec={nd} pre={len(d) - nd}"
        f.write(f"{t1:.4f} {kind} ms={(t1 - t0) * 1000:.1f} gap={gap:.1f}{extra}\\n")
''')
m = 'logger = init_logger(__name__)\n'; assert s.count(m) == 1
s = s.replace(m, m + '_GLM5_STAGE_TIMING = os.environ.get("GLM5_STAGE_TIMING") == "1"  # [claude-stt-01]\n')
open(p, "w").write(s); print("edited", p)
