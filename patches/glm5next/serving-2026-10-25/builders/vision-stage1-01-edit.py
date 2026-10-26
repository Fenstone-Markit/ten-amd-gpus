#!/usr/bin/env python3
# vision-stage1-01-edit.py <patch dir> [claude-vis-01]: GLM5_VISION_FIRST_STAGE_ONLY=1 (off unless set).
# The vision tower (347 tensors, 1.05 GiB bf16) is built by Glm5NextForConditionalGeneration on EVERY pipeline stage, split
# 4 ways (about 0.26 GiB per card), but vLLM only runs the image encoder on the first stage (gpu_model_runner: "if
# self.supports_mm_inputs and is_first_rank"). With the switch on, stages 2 and 3 get vLLM's StageMissingLayer placeholder
# for it (the same placeholder --language-model-only uses: no weights, never called), and profile_run skips the
# dummy-image encoder pass on those stages (it would call the placeholder). Nothing changes on stage 1 or with the switch off.
import sys
d = sys.argv[1]
p = d + "/vllm/models/glm5next/nvidia/model.py"; s = open(p).read()
a = '''        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
'''
assert s.count(a) == 1
s = s.replace(a, '''        import contextlib as _glm5_cl, os as _glm5_os  # [claude-vis-01]
        from vllm.model_executor.models.utils import StageMissingLayer as _glm5_SML, no_init_weights as _glm5_niw
        _glm5_vis_skip = (_glm5_os.environ.get("GLM5_VISION_FIRST_STAGE_ONLY") == "1" and not get_pp_group().is_first_rank)
        with self._mark_tower_model(vllm_config, {"image", "video"}), (
            _glm5_niw(self, lambda mod: _glm5_SML("vision_tower", mod)) if _glm5_vis_skip else _glm5_cl.nullcontext()
        ):
            self.visual = Glm5NextVisionTransformer(
''')
open(p, "w").write(s); print("edited", p)
p = d + "/vllm/v1/worker/gpu_model_runner.py"; s = open(p).read()
a = '''        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
'''
assert s.count(a) == 1
s = s.replace(a, '''        # Profile with multimodal encoder & encoder cache.
        # [claude-vis-01] GLM5_VISION_FIRST_STAGE_ONLY=1: later stages hold no vision tower and never run the encoder.
        if self.supports_mm_inputs and (os.environ.get("GLM5_VISION_FIRST_STAGE_ONLY") != "1" or get_pp_group().is_first_rank):
            mm_config = self.model_config.multimodal_config
''')
open(p, "w").write(s); print("edited", p)
