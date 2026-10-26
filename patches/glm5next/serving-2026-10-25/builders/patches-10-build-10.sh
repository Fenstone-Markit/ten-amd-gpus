#!/bin/bash
# patches-10-build-10: node02. ~/glm-serve/claude-patches-10 = claude-patches-09 plus two MTP-with-pipeline fixes:
#  1. models/glm5next/nvidia/mtp.py (from the overlay): Glm5NextMTP declares SupportsPP. glm-serve-37-01 stopped at config time
#     ("Pipeline parallelism is not supported for this model"). vLLM builds the drafter only on the last stage; its forward already
#     accepts intermediate_tensors; the in-image Glm4MoeLiteMTP does exactly this.
#  2. v1/worker/gpu_model_runner.py: six places that use self.drafter now also require that this stage has one
#     (hasattr(self, "drafter")). The drafter exists only on the last stage (line 591), but the dummy/profile run, the attention
#     metadata build and the drafter attention set-up ran on every stage: glm-serve-37-02 died in profile_run on stages 1-5 with
#     "'GPUModelRunner' object has no attribute 'drafter'". On the last stage nothing changes.
#  3. v1/worker/gpu_worker.py: with GLM5_TB_SIGNAL=1, every worker appends all its Python thread stacks to
#     /cache/tb/tb-<pid>.txt every 90 s (faulthandler.dump_traceback_later: a C watchdog thread, needs no GIL and no signal;
#     build-03's SIGUSR1 handler wrote nothing in 37-05). glm-serve-37-04 hung on its first decode steps with every card
#     idle (stage 1 waiting for work, stages 2-6 blocked); ptrace is not allowed in the container, so this is how to see where.
#     Off unless the variable is set.
#  4. v1/engine/core.py (from the overlay): with a draft model and without async scheduling, one batch in flight at a time
#     (batch queue size 1 instead of the 6 that pipeline parallelism asks for). glm-serve-37-06's stack dumps: stage 1 idle
#     waiting for work, stages 2-6 waiting to receive a step's hidden states that stage 1 never sent. Cause, from the code: with
#     a batch queue, the engine fetches the step's draft token BEFORE it records the step's sampled token, so the scheduler sees
#     one "new" token (the stale draft) and schedules a step the stages disagree on. One batch at a time keeps the order
#     schedule -> run -> record sampled token -> fetch draft. With one request decode is sequential anyway.
#  5. mtp.py load_weights: also load the main model's embedding (model.embed_tokens) into the draft model. With pipeline
#     stages vLLM does not share the target's embedding with the drafter (the last stage has none: llm_base_proposer
#     _maybe_share_embeddings runs only when the pipeline size is 1), and GLM's draft layer carries no embedding copy of its
#     own, so the loader skipped it (no layer index) and the drafter ran on an uninitialised embedding: glm-serve-37-07
#     served correct answers but accepted only 11-18% of drafts (the RDNA4 port reports 91-97%).
#  6. v1/engine/core.py: GLM5_STEP_TIMING=1 logs, every 64 decode steps, the average time per step spent in: scheduling, the
#     model run (all stages), sampling (+ drafting on the last stage), the scheduler update, and fetching the drafts. Read-only
#     timing (no GPU syncs added); off unless the variable is set. To find where MTP's extra ~38 ms per step goes.
#  7. v1/worker/gpu_model_runner.py (claude-mtp-07): after drafting, the last stage sends every earlier stage, per request,
#     [next input token, valid sampled count, accepted count, draft tokens]; earlier stages apply them as the last stage does.
#     (a) Lets async scheduling work with MTP across stages (vLLM 0.27 sent one token per request and asserted). (b) Fixes the
#     earlier stages' KDA (linear-attention) layers: only the last stage ran _update_states_after_model_execute, so stages 1-5
#     always assumed one accepted token and dropped the second token's state whenever a draft was accepted.
#     (c, build-09) earlier stages in async mode also trim the optimistic -1 placeholders the way the last stage does; without it
#     glm-serve-37-11 fed stage 1 wrong tokens ('gateK 17547' for K1757).
#  8. v1/engine/core.py (claude-mtp-08, replaces build-05's batch queue of 1): the batch queue stays at 6 so prompt chunks
#     still overlap across stages (build-05 made a 32k prompt take 41 s instead of 10 s); while any request is decoding and a
#     batch is in flight no new batch is scheduled, and drafts are fetched right after a step's output is recorded, only when
#     nothing is in flight (so the scheduler never sees a draft before the token it follows).
# Refuses if any anchor is missing or not unique.
set -u
P9=~/glm-serve/claude-patches-09; P10=~/glm-serve/claude-patches-10
SRC=~/sandbox/workspace/glm5next-port-01/overlay/app/vllm/vllm/models/glm5next/nvidia/mtp.py
no() { echo "REFUSED: $*"; exit 1; }
[ -f $SRC ] || no "no overlay mtp.py"
rm -rf $P10 && cp -a $P9 $P10 || no "copy failed"
cp $SRC $P10/vllm/models/glm5next/nvidia/mtp.py
cp $P10/vllm/v1/worker/gpu_model_runner.py /tmp/gmr-p09.py
python3 - $P10/vllm/models/glm5next/nvidia/mtp.py $P10/vllm/v1/worker/gpu_model_runner.py <<'PY' || no "edit failed"
import sys
p = sys.argv[1]; s = open(p).read()
def rep(s, a, b):
    assert s.count(a) == 1, (a[:60], s.count(a)); return s.replace(a, b, 1)
s = rep(s, "class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):",
        "class Glm5NextMTP(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):  # [claude-mtp-01] PP: drafter on the last stage")
b = "from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix"
s = rep(s, b, b + "\nfrom vllm.model_executor.models.interfaces import SupportsPP  # [claude-mtp-01]")
a = """            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
"""
s = rep(s, a, """            # [claude-mtp-05] pipeline stages: the drafter gets no shared embedding, so load the main one into it
            if name == "model.embed_tokens.weight" and name in params_dict:
                _p = params_dict[name]
                getattr(_p, "weight_loader", default_weight_loader)(_p, loaded_weight)
                loaded_params.add(name)
                continue
""" + a)
assert "default_weight_loader" in s.split("class Glm5NextMultiTokenPredictorLayer")[0], "default_weight_loader not imported"
open(p, "w").write(s); print("mtp.py: Glm5NextMTP declares SupportsPP; loads the main embedding")
p = sys.argv[2]; s = open(p).read()
H = 'hasattr(self, "drafter")'
edits = [
 ("            if self.speculative_config and spec_decode_common_attn_metadata is None:\n",
  f"            if self.speculative_config and {H} and spec_decode_common_attn_metadata is None:  # [claude-mtp-02]\n"),
 ("            if self.speculative_config and isinstance(\n                self.drafter, (Step3p5MTPProposer, Glm5NextMTPProposer)\n",
  f"            if self.speculative_config and {H} and isinstance(  # [claude-mtp-02]\n                self.drafter, (Step3p5MTPProposer, Glm5NextMTPProposer)\n"),
 ("            elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):\n",
  f"            elif self.speculative_config and {H} and isinstance(self.drafter, Gemma4Proposer):  # [claude-mtp-02]\n"),
 ("            if self.speculative_config and (\n                self.speculative_config.use_eagle()\n                or self.speculative_config.uses_draft_model()\n                or self.speculative_config.uses_extract_hidden_states()\n            ):\n                assert isinstance(\n                    self.drafter,\n",
  f"            if self.speculative_config and {H} and (  # [claude-mtp-02]\n                self.speculative_config.use_eagle()\n                or self.speculative_config.uses_draft_model()\n                or self.speculative_config.uses_extract_hidden_states()\n            ):\n                assert isinstance(\n                    self.drafter,\n"),
 ("        if self.speculative_config and (\n            self.speculative_config.use_eagle()\n            or self.speculative_config.uses_draft_model()\n        ):\n",
  f"        if self.speculative_config and {H} and (  # [claude-mtp-02]\n            self.speculative_config.use_eagle()\n            or self.speculative_config.uses_draft_model()\n        ):\n"),
 ("        if self.speculative_config and (\n            self.speculative_config.use_eagle()\n            or self.speculative_config.uses_draft_model()\n            or self.speculative_config.uses_extract_hidden_states()\n        ):\n            assert isinstance(\n                self.drafter,\n",
  f"        if self.speculative_config and {H} and (  # [claude-mtp-02]\n            self.speculative_config.use_eagle()\n            or self.speculative_config.uses_draft_model()\n            or self.speculative_config.uses_extract_hidden_states()\n        ):\n            assert isinstance(\n                self.drafter,\n"),
]
for a, b in edits: s = rep(s, a, b)
open(p, "w").write(s); print(f"gpu_model_runner.py: {len(edits)} drafter uses guarded")
PY
python3 - $P10/vllm/v1/worker/gpu_worker.py <<'PY' || no "gpu_worker.py edit failed"
import sys
p = sys.argv[1]; s = open(p).read()
a = "import gc\nimport os\nimport time\n"
assert s.count(a) == 1, s.count(a)
b = a + """if os.environ.get("GLM5_TB_SIGNAL") == "1":  # [claude-mtp-03] stack dump every 90 s (debug)
    import faulthandler as _glm5_fh, signal as _glm5_sig
    os.makedirs("/cache/tb", exist_ok=True)
    _glm5_tb_file = open(f"/cache/tb/tb-{os.getpid()}.txt", "w")
    _glm5_fh.dump_traceback_later(90, repeat=True, file=_glm5_tb_file)
"""
open(p, "w").write(s.replace(a, b, 1)); print("gpu_worker.py: stack dump every 90 s behind GLM5_TB_SIGNAL=1")
PY
python3 ~/fp8-rdna3-01/mtp07-edit.py $P10/vllm/v1/worker/gpu_model_runner.py || no "mtp-07 edit failed"
CORE=~/sandbox/workspace/glm5next-port-01/overlay/app/vllm/vllm/v1/engine/core.py
mkdir -p $P10/vllm/v1/engine && cp $CORE $P10/vllm/v1/engine/core.py
python3 - $P10/vllm/v1/engine/core.py <<'PY' || no "core.py edit failed"
import sys
p = sys.argv[1]; s = open(p).read()
a = "        self.batch_queue_size = vllm_config.max_concurrent_batches\n"
assert s.count(a) == 1, s.count(a)
b = a + """        if (  # [claude-mtp-04] draft model + no async scheduling: one batch in flight (keeps sampled-before-draft order)
            self.use_spec_decode
            and not vllm_config.scheduler_config.async_scheduling
            and self.batch_queue_size > 1
        ):
            logger.info("[claude-mtp-04] speculative decoding without async scheduling: batch queue %d -> 1",
                        self.batch_queue_size)
            self.batch_queue_size = 1
"""
assert s.find("self.use_spec_decode = ") < s.find(a), "use_spec_decode is set after the batch queue"
s = s.replace(a, b, 1)
# [claude-mtp-06] step timing
st_a = """        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
"""
assert s.count(st_a) == 1, ("step anchor", s.count(st_a))
st_b = """        _t0 = time.perf_counter()  # [claude-mtp-06]
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        _t1 = time.perf_counter()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            _t2 = time.perf_counter()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
            _t3 = time.perf_counter()

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        if _GLM5_STEP_TIMING and scheduler_output.total_num_scheduled_tokens <= 16:
            _glm5_step_add(self, "schedule", _t1 - _t0, "run", _t2 - _t1, "sample+draft", _t3 - _t2,
                           "update", time.perf_counter() - _t3)
"""
s = s.replace(st_a, st_b, 1)
ps_a = """            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)
"""
assert s.count(ps_a) == 1, ("post_step anchor", s.count(ps_a))
ps_b = """            _t0 = time.perf_counter()  # [claude-mtp-06]
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)
            if _GLM5_STEP_TIMING:
                _glm5_step_add(self, "fetch-drafts", time.perf_counter() - _t0)
"""
s = s.replace(ps_a, ps_b, 1)
helper = """
_GLM5_STEP_TIMING = os.environ.get("GLM5_STEP_TIMING") == "1"  # [claude-mtp-06]


def _glm5_step_add(core, *kv):
    acc = core.__dict__.setdefault("_glm5_step_acc", {})
    for k, v in zip(kv[0::2], kv[1::2]):
        acc[k] = acc.get(k, 0.0) + v
    if kv[0] == "schedule":
        core.__dict__["_glm5_step_n"] = core.__dict__.get("_glm5_step_n", 0) + 1
        n = core._glm5_step_n
        if n % 64 == 0:
            logger.info("[claude-mtp-06] last 64 decode steps, ms per step: %s",
                        ", ".join(f"{k} {1000 * v / 64:.1f}" for k, v in acc.items()))
            acc.clear()

"""
anchor = "\nclass EngineCore:"
assert s.count(anchor) == 1, ("class anchor", s.count(anchor))
s = s.replace(anchor, helper + anchor, 1)
assert "\nimport time\n" in s and "\nimport os\n" in s, "core.py lacks import os/time"
open(p, "w").write(s); print("core.py: one batch in flight with a draft model and no async scheduling; step timing behind GLM5_STEP_TIMING=1")
PY
python3 ~/fp8-rdna3-01/mtp08-core-edit.py $P10/vllm/v1/engine/core.py || no "mtp-08 edit failed"
for f in vllm/models/glm5next/nvidia/mtp.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu_worker.py vllm/v1/engine/core.py; do python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" $P10/$f || no "syntax $f"; done
(cd $P10 && diff -u $SRC vllm/models/glm5next/nvidia/mtp.py > vllm/models/glm5next/nvidia/mtp.py.diff; diff -u /tmp/gmr-p09.py vllm/v1/worker/gpu_model_runner.py > vllm/v1/worker/gpu_model_runner.py.mtp.diff)
echo "claude-patches-10: $(find $P10/vllm -name '*.py' | wc -l) files"; (cd $P10 && md5sum vllm/models/glm5next/nvidia/mtp.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu_worker.py vllm/v1/engine/core.py)
grep -c "claude-mtp-02" $P10/vllm/v1/worker/gpu_model_runner.py
