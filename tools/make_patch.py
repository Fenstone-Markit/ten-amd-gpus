"""Make the RDNA3 W4A16 patch. Runs on the host with plain python3; touches no container.

Input:  ~/handtest/src/rdna3_w4a16.py          (copied from the image; must match production's hash)
Output: ~/handtest/src/rdna3_w4a16_patched.py  (to be mounted over the original in the experiment only)

What the patch does, and why:
  1. import replace_parameter (the same helper the Triton kernel uses)
  2. accept uint4 (asymmetric, explicit zero points) as well as uint4b8
  3. when the checkpoint supplies zero points: accept ONLY the compressed-tensors layout
     [N//8, K//G] packed on dim 0, and transpose it to the kernel's [K//G, N//8]
     (identical to what TritonW4A16LinearKernel does). Any other layout raises at load
     time, loudly, because GPTQ and AWQ store zeros with different conventions.
  4. call the kernel with use_v2_format = True exactly when the checkpoint supplies zero
     points, so the kernel does not add GPTQ's +1 to true zeros. Symmetric checkpoints
     (no zero points) keep False and behave exactly as before.
"""
import difflib, hashlib, os, py_compile, sys

SRC = os.path.expanduser("~/handtest/src/rdna3_w4a16.py")
DST = os.path.expanduser("~/handtest/src/rdna3_w4a16_patched.py")
WANT_MD5 = "3b421ed1e3e6fdf9d2dd937413a40774"

orig = open(SRC).read()
md5 = hashlib.md5(orig.encode()).hexdigest()
if md5 != WANT_MD5:
    sys.exit(f"ABORT: source md5 {md5} is not production's {WANT_MD5}")

EDITS = [
    ("import",
     "from vllm.scalar_type import scalar_types\n",
     "from vllm.scalar_type import scalar_types\n"
     "from vllm.model_executor.layers.quantization.utils import replace_parameter  # [fenstone patch]\n"),
    ("types",
     "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]\n",
     "    # [fenstone patch] uint4 added: asymmetric compressed-tensors checkpoints with explicit zeros\n"
     "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]\n"),
    ("zero points",
     "layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)\n            )\n",
     "layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)\n            )\n"
     "        else:\n"
     "            # [fenstone patch] The checkpoint supplies its own zero points. Accept only the\n"
     "            # compressed-tensors layout [N//8, K//G], N packed on dim 0, and transpose it to\n"
     "            # the kernel's [K//G, N//8] (same as TritonW4A16LinearKernel). GPTQ and AWQ store\n"
     "            # zeros with other conventions and must fail here rather than dequantise wrongly.\n"
     "            zp = getattr(layer, self.w_zp_name)\n"
     "            groups = c.partition_weight_shape[0] // c.group_size\n"
     "            n8 = c.partition_weight_shape[1] // 8\n"
     "            packed_dim = getattr(zp, \"packed_dim\", None)\n"
     "            if tuple(zp.shape) != (n8, groups) or packed_dim != 0:\n"
     "                raise NotImplementedError(\n"
     "                    f\"RDNA3 W4A16 [fenstone patch]: zero points {tuple(zp.shape)}, packed_dim \"\n"
     "                    f\"{packed_dim}, are not the compressed-tensors layout ({n8}, {groups}) on dim 0\"\n"
     "                )\n"
     "            replace_parameter(\n"
     "                layer,\n"
     "                self.w_zp_name,\n"
     "                torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),\n"
     "            )\n"),
    ("apply",
     "output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx, False)\n",
     "# [fenstone patch] explicit zero points are true zeros: no GPTQ +1 (use_v2_format)\n"
     "        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx, c.zero_points)\n"),
]

text = orig
for name, old, new in EDITS:
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT: edit '{name}' anchor found {n} times, expected exactly 1; nothing written")
    text = text.replace(old, new)

open(DST, "w").write(text)
try:
    py_compile.compile(DST, doraise=True)
except py_compile.PyCompileError as exc:
    os.remove(DST)
    sys.exit(f"ABORT: patched file does not compile, removed: {exc}")

print(f"source md5 {md5} (production's)")
print(f"wrote {DST}, md5 {hashlib.md5(text.encode()).hexdigest()}, compiles OK")
print("=== diff (read this before anything is mounted anywhere)")
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), text.splitlines(True),
                                           "rdna3_w4a16.py (production)", "rdna3_w4a16_patched.py"))
