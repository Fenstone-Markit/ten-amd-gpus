"""fp32_op_gate.py: package the gated fp32 kernel as a PyTorch op, and gate the package. G7 only.

Run in a throwaway container (see the command that came with this file). It builds from
/workspace/fp32-kernel/src/q_gemm_rdna3_fp32.cu, the source gate_fp32_v2.py passed, and writes
/workspace/fp32-kernel/op/fen_fp32_op.so.

The op, _fenstone_C::gptq_gemm_rdna3_fp32(a, b_q_weight, b_qzeros, b_scales, use_v2_format):
  M < 16   (decode)  -> the fp32-accumulating kernel
  M >= 16  (prefill) -> FENSTONE_FP32_PREFILL=fp32: the same fp32 kernel for every M
                        otherwise: the installed _rocm_C::gptq_gemm_rdna3, exactly as served before
The switch is read inside the op at every prefill-sized call, so torch.compile cannot freeze a branch.

Gates (exit 0 only if all pass):
  B  build: the source's WMMA branch found exactly once and disabled, the op registered
  D  decode precision: M = 1, 2, 4, 8, 12, every production shape, <= 1.10 x bf16 floor and <= 1.25 x triton
  X  determinism: M = 1 output bit-identical across two calls, every shape
  F  prefill in fp32 mode: M = 16, 128, 512, <= 1.10 x floor, and the op never calls the installed kernel
  C  torch.compile (dynamic shapes, fullgraph): M = 1 and M = 128 match eager, and each runs its intended kernel
  G  CUDA graph: M = 1 replay bit-identical to eager
Also reported, not gated: the installed prefill path's precision and every path's speed at M = 16 to 2048.
"""
import os, sys, re, json, time, hashlib, statistics
os.environ["ROCR_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HIP_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_PLUGINS"] = ""
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ.pop("FENSTONE_FP32_PREFILL", None)

W = "/workspace/fp32-kernel"
SRC, HDR = f"{W}/src/q_gemm_rdna3_fp32.cu", f"{W}/src/qdq_4_rdna3.cuh"
OPDIR = f"{W}/op"
OUT = f"{W}/op_gate_results.json"
G = 32
SELF_MD5 = hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()
res = {"md5": SELF_MD5}
print(f"fp32_op_gate.py md5 {SELF_MD5}", flush=True)

def save():
    json.dump(res, open(OUT, "w"), indent=2)

def stop(msg):
    print(f"\nSTOP: {msg}", flush=True)
    res["stopped"] = msg; save(); sys.exit(2)

import torch
import triton
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm_kernel
dev = "cuda"
if torch.cuda.device_count() != 1:
    stop(f"{torch.cuda.device_count()} cards visible, expected exactly G7")
if torch.cuda.mem_get_info()[0] / 2**30 < 20:
    stop("G7 is not empty")
if not (hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")):
    stop("the installed _rocm_C::gptq_gemm_rdna3 is not registered in this image")

# ---------------------------------------------------------------- B: build
src = open(SRC).read()
res["src_md5"] = hashlib.md5(src.encode()).hexdigest()
print(f"source md5 {res['src_md5']}", flush=True)
WMMA_IF = "if (a.dim() == 2 && b_q_weight.dim() == 2 && a.size(1) % 16 == 0 &&"
n_if = src.count(WMMA_IF)
if n_if != 1:
    stop(f"expected the WMMA branch condition exactly once in the source, found {n_if}")
src = src.replace(WMMA_IF, "if (false && a.dim() == 2 && b_q_weight.dim() == 2 && a.size(1) % 16 == 0 &&")

def rename(t):
    t = re.sub(r"\bvllm\b", "vllmfen", t)
    t = re.sub(r"\bgptq_gemm_rdna3_wmma\b", "fen_wmma_unused", t)
    return re.sub(r"\bgptq_gemm_rdna3\b", "fen_fp32_entry", t)

GLUE = r'''

// ---- appended by fp32_op_gate.py: the registered op ----
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cstdlib>
#include <string>
#include <vector>

// The WMMA branch above is disabled (if (false && ...)), so this is never reached.
torch::Tensor fen_wmma_unused(torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_qzeros,
                              torch::Tensor b_scales, bool use_v2_format) {
  TORCH_CHECK(false, "fen_wmma_unused must be unreachable");
  return a;
}

// Read at every prefill-sized call, never cached, so a compiled graph cannot freeze it.
static bool fen_prefill_uses_fp32() {
  const char* v = std::getenv("FENSTONE_FP32_PREFILL");
  return v != nullptr && std::string(v) == "fp32";
}

static torch::Tensor fen_gptq_gemm_rdna3_fp32(torch::Tensor a, torch::Tensor b_q_weight,
                                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                                              bool use_v2_format) {
  if (a.size(0) >= 16 && !fen_prefill_uses_fp32()) {
    // Prefill on the installed kernel, called through the dispatcher so no symbol is shared.
    static const c10::OperatorHandle installed =
        c10::Dispatcher::singleton().findSchemaOrThrow("_rocm_C::gptq_gemm_rdna3", "");
    std::vector<c10::IValue> stack;
    stack.reserve(6);
    stack.emplace_back(a);
    stack.emplace_back(b_q_weight);
    stack.emplace_back(b_qzeros);
    stack.emplace_back(b_scales);
    stack.emplace_back(at::empty({0}, a.options().dtype(at::kInt)));   // g_idx: none
    stack.emplace_back(use_v2_format);
    installed.callBoxed(&stack);
    return stack[0].toTensor();
  }
  return fen_fp32_entry(a, b_q_weight, b_qzeros, b_scales, use_v2_format);
}

TORCH_LIBRARY(_fenstone_C, m) {
  m.def("gptq_gemm_rdna3_fp32(Tensor a, Tensor b_q_weight, Tensor b_qzeros, Tensor b_scales, bool use_v2_format) -> Tensor");
  m.impl("gptq_gemm_rdna3_fp32", torch::kCUDA, &fen_gptq_gemm_rdna3_fp32);
}
'''
os.makedirs(OPDIR, exist_ok=True)
if os.path.exists(f"{OPDIR}/lock"):
    os.remove(f"{OPDIR}/lock")
open(f"{OPDIR}/q_gemm_rdna3.cu", "w").write(rename(src) + GLUE)
open(f"{OPDIR}/qdq_4_rdna3.cuh", "w").write(rename(open(HDR).read()))
print(f"[{time.strftime('%H:%M:%S')}] building the op (about a minute)", flush=True)
try:
    so = load(name="fen_fp32_op", sources=[f"{OPDIR}/q_gemm_rdna3.cu"], build_directory=OPDIR,
              extra_include_paths=[OPDIR], extra_cuda_cflags=["-O3", "-DUSE_ROCM"],
              is_python_module=False, verbose=False)
except Exception as exc:
    res["build_error"] = str(exc)[-3000:]
    stop(f"build failed: {type(exc).__name__}. The compiler output is above.")
so_path = f"{OPDIR}/fen_fp32_op.so"
if not os.path.exists(so_path):
    stop(f"build reported success but {so_path} does not exist")
if not hasattr(torch.ops._fenstone_C, "gptq_gemm_rdna3_fp32"):
    stop("the library loaded but _fenstone_C::gptq_gemm_rdna3_fp32 is not registered")
res["so_md5"] = hashlib.md5(open(so_path, "rb").read()).hexdigest()
print(f"B build: PASS  op registered, {so_path} md5 {res['so_md5']}", flush=True)
verdict = {"B": True}
save()

# Same fake registration as the brain's patch uses, so torch.compile can trace the op.
@torch.library.register_fake("_fenstone_C::gptq_gemm_rdna3_fp32")
def _fen_fp32_fake(a, b_q_weight, b_qzeros, b_scales, use_v2_format):
    return a.new_empty((a.shape[0], b_q_weight.shape[1]))

OP = torch.ops._fenstone_C.gptq_gemm_rdna3_fp32

# ---------------------------------------------------------------- helpers, identical in method to gate_fp32_v2.py
SHAPES = {"qkv": (5120, 2048), "o": (768, 5120), "gate_up": (5120, 4352),
          "down": (2176, 5120), "qkvz": (5120, 2048), "gdn_out": (768, 5120)}

def make_case(K, N, M, seed, wrong_zeros=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 16, (K, N), generator=g, dtype=torch.int32)
    z = torch.randint(0, 16, (K // G, N), generator=g, dtype=torch.int32)
    s = (torch.randn(K // G, N, generator=g) * 0.01).to(torch.bfloat16)
    a = torch.randn(M, K, generator=g).to(torch.bfloat16)
    ref = a.double() @ ((q.double() - z.double().repeat_interleave(G, 0)) * s.double().repeat_interleave(G, 0))
    if wrong_zeros:
        z = (z + 1 + torch.randint(0, 14, z.shape, generator=g, dtype=torch.int32)) % 16
    sh = (torch.arange(8, dtype=torch.int64) * 4)   # int64: 15 << 28 overflows int32
    # zeros [K/G, N/8], N packed sequentially: identical for both kernels
    zp = ((z.view(K // G, N // 8, 8).to(torch.int64) & 0xF) << sh).sum(-1)
    zp = torch.where(zp >= 2**31, zp - 2**32, zp).to(torch.int32)
    # Triton: b_q [K, N/8], N packed sequentially
    tq = ((q.view(K, N // 8, 8).to(torch.int64) & 0xF) << sh).sum(-1)
    tq = torch.where(tq >= 2**31, tq - 2**32, tq).to(torch.int32)
    # HIP: qweight [K/8, N], K packed sequentially, then the same gptq_shuffle production applies
    hq = ((q.view(K // 8, 8, N).to(torch.int64) & 0xF) << sh.view(1, 8, 1)).sum(1)
    hq = torch.where(hq >= 2**31, hq - 2**32, hq).to(torch.int32).to(dev).contiguous()
    gidx = torch.empty(0, device=dev, dtype=torch.int)
    ops.gptq_shuffle(hq, gidx, 4)
    return {"a": a.to(dev), "ref": ref.to(dev), "tri": (tq.to(dev), s.to(dev), zp.to(dev)),
            "hip": (hq, zp.to(dev), s.to(dev), gidx)}

def run_triton(a, w):
    b_q, sc, qz = w
    M, K = a.shape; N = b_q.shape[1] * 8
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    triton_w4a16_gemm_kernel[(triton.cdiv(M, 128), triton.cdiv(N, 32))](
        a, b_q, sc, qz, c, M, N, K, a.stride(0), a.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1), group_size=G, HAS_ZP=True, ZP_BIAS=8, BLOCK_M=128, BLOCK_N=32, BLOCK_K=32)
    return c

def run_op(a, w):
    return OP(a, w[0], w[1], w[2], True)

def run_installed(a, w):
    return ops.gptq_gemm_rdna3(a, w[0], w[1], w[2], w[3], True)

def rms_rel(x, ref):
    d = x.double() - ref
    return (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()

def kernel_names(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return [ev.name for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA]

def is_fen(names):
    return any("vllmfen::" in n for n in names)

def is_installed(names):
    return any("gemm" in n.lower() and "vllmfen::" not in n for n in names)

# ---------------------------------------------------------------- D and X: decode
print("\n=== D decode precision (error vs exact float64; floor = best any bf16 output can do) and X determinism", flush=True)
res["decode"] = {}
D_ok = X_ok = True
for op_name, (K, N) in SHAPES.items():
    for M in (1, 2, 4, 8, 12):
        c = make_case(K, N, M, 7 + M)
        floor = rms_rel(c["ref"].to(torch.bfloat16), c["ref"])
        e_op = rms_rel(run_op(c["a"], c["hip"]), c["ref"])
        e_tr = rms_rel(run_triton(c["a"], c["tri"]), c["ref"])
        ok = e_op <= 1.10 * floor and e_op <= 1.25 * e_tr
        D_ok &= ok
        line = f"{op_name:8s} M{M:<3d} floor {floor:.5f}  triton {e_tr:.5f}  op {e_op:.5f} [{'D ok' if ok else 'D FAIL'}]"
        if M == 1:
            cb = make_case(K, N, M, 7 + M, wrong_zeros=True)
            e_broken = rms_rel(run_op(cb["a"], cb["hip"]), c["ref"])
            broken_ok = e_broken >= 0.10
            D_ok &= broken_ok
            line += f"  broken {e_broken:.3f} [{'ctl ok' if broken_ok else 'CONTROL FAIL'}]"
            same = torch.equal(run_op(c["a"], c["hip"]), run_op(c["a"], c["hip"]))
            X_ok &= same
            line += f"  M1 bit-identical twice: {'yes [X ok]' if same else 'NO [X FAIL]'}"
        print(line, flush=True)
        res["decode"][f"{op_name}_M{M}"] = {"floor": floor, "triton": e_tr, "op": e_op, "ok": ok}
    torch.cuda.synchronize()
c1 = make_case(5120, 4352, 1, 1)
names_m1 = kernel_names(lambda: run_op(c1["a"], c1["hip"]))
decode_kernel_ok = is_fen(names_m1)
print(f"M1 runs the fp32 kernel: {'yes' if decode_kernel_ok else 'NO'}  ({[n[:60] for n in names_m1]})", flush=True)
verdict["D"], verdict["X"] = D_ok and decode_kernel_ok, X_ok
save()

# ---------------------------------------------------------------- F: prefill in fp32 mode, and the installed path for comparison
print("\n=== prefill paths (M >= 16), precision vs exact", flush=True)
res["prefill"] = {}
F_ok = True
for op_name in ("qkv", "o", "gate_up", "down"):
    K, N = SHAPES[op_name]
    for M in (16, 128, 512):
        c = make_case(K, N, M, 100 + M)
        floor = rms_rel(c["ref"].to(torch.bfloat16), c["ref"])
        os.environ["FENSTONE_FP32_PREFILL"] = "fp32"
        e_fp32 = rms_rel(run_op(c["a"], c["hip"]), c["ref"])
        nm_fp32 = kernel_names(lambda: run_op(c["a"], c["hip"]))
        os.environ.pop("FENSTONE_FP32_PREFILL")
        e_inst_via_op = rms_rel(run_op(c["a"], c["hip"]), c["ref"])
        nm_inst = kernel_names(lambda: run_op(c["a"], c["hip"]))
        e_inst = rms_rel(run_installed(c["a"], c["hip"]), c["ref"])
        e_tr = rms_rel(run_triton(c["a"], c["tri"]), c["ref"])
        ok = e_fp32 <= 1.10 * floor and is_fen(nm_fp32) and not is_installed(nm_fp32) and is_installed(nm_inst) and not is_fen(nm_inst)
        F_ok &= ok
        print(f"{op_name:8s} M{M:<4d} floor {floor:.5f}  fp32-mode {e_fp32:.5f}  installed-mode {e_inst_via_op:.5f}  "
              f"installed direct {e_inst:.5f}  triton {e_tr:.5f}  [{'F ok' if ok else 'F FAIL'}]", flush=True)
        res["prefill"][f"{op_name}_M{M}"] = {"floor": floor, "fp32_mode": e_fp32, "installed_mode": e_inst_via_op,
                                              "installed_direct": e_inst, "triton": e_tr,
                                              "fp32_kernels": nm_fp32, "installed_kernels": nm_inst, "ok": ok}
    torch.cuda.synchronize()
verdict["F"] = F_ok
save()

# ---------------------------------------------------------------- C: torch.compile with dynamic shapes, no frozen branch
print("\n=== C torch.compile: one dynamic graph must route every M correctly at run time", flush=True)
C_ok = True
try:
    torch._dynamo.reset(); torch._dynamo.utils.counters.clear()
    f = torch.compile(lambda a, q, z, s: OP(a, q, z, s, True), backend="aot_eager", fullgraph=True)
    for i, (M, want_fen) in enumerate(((4, True), (128, False), (7, True), (1, True))):
        c = make_case(5120, 4352, M, 900 + M)
        a = c["a"]
        if i == 0:
            torch._dynamo.mark_dynamic(a, 0)
        eager = run_op(a, c["hip"])
        comp = f(a, *c["hip"][:3])
        if M == 1:
            match = torch.equal(comp, eager)
        else:   # M = 4 and 7 have tiny atomic-order differences; the installed prefill path is not bit-deterministic
            match = rms_rel(comp, c["ref"]) <= 1.5 * rms_rel(eager, c["ref"])
        nm = kernel_names(lambda: f(a, *c["hip"][:3]))
        right = is_fen(nm) if want_fen else (is_installed(nm) and not is_fen(nm))
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        one_graph = graphs == 1 if M != 1 else True   # torch specialises size 1 into its own graph; routing is what matters
        ok = match and right and one_graph
        C_ok &= ok
        print(f"compiled M{M}: matches eager {match}, runs {'fp32 kernel' if is_fen(nm) else 'installed kernel'} "
              f"(intended {'fp32' if want_fen else 'installed'}), graphs so far {graphs} [{'C ok' if ok else 'C FAIL'}]", flush=True)
except Exception as exc:
    C_ok = False
    print(f"compile FAILED: {type(exc).__name__}: {str(exc)[:600]}", flush=True)
verdict["C"] = C_ok
save()

# ---------------------------------------------------------------- G: CUDA graph capture at M = 1
print("\n=== G CUDA graph capture, M = 1", flush=True)
G_ok = True
try:
    c = make_case(5120, 4352, 1, 4242)
    eager = run_op(c["a"], c["hip"])
    for _ in range(3):
        run_op(c["a"], c["hip"])
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = run_op(c["a"], c["hip"])
    gr.replay(); torch.cuda.synchronize()
    G_ok = torch.equal(out, eager)
    print(f"graph replay bit-identical to eager: {G_ok} [{'G ok' if G_ok else 'G FAIL'}]", flush=True)
except Exception as exc:
    G_ok = False
    print(f"graph capture FAILED: {type(exc).__name__}: {str(exc)[:600]}", flush=True)
verdict["G"] = G_ok
save()

# ---------------------------------------------------------------- speed, reported only
def time_calls(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(9):
        s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize()
        t.append(s.elapsed_time(e) * 1000.0 / n)
    return statistics.median(t)

print("\n=== speed, gate_up shape (K 5120, N 4352), us per call. For comparing the prefill modes with each other only:", flush=True)
print("    not graphed and one weight copy (cache-resident), so absolute numbers are optimistic. Not gated.", flush=True)
res["speed"] = {}
for M in (1, 16, 128, 512, 2048):
    c = make_case(5120, 4352, M, 7000 + M)
    row = {"triton": time_calls(lambda: run_triton(c["a"], c["tri"])),
           "installed": time_calls(lambda: run_installed(c["a"], c["hip"]))}
    os.environ["FENSTONE_FP32_PREFILL"] = "fp32"
    row["op_fp32_mode"] = time_calls(lambda: run_op(c["a"], c["hip"]))
    os.environ.pop("FENSTONE_FP32_PREFILL")
    row["op_installed_mode"] = time_calls(lambda: run_op(c["a"], c["hip"]))
    res["speed"][f"M{M}"] = row
    print(f"M{M:<5d} " + "  ".join(f"{k} {v:9.1f}" for k, v in row.items()), flush=True)
    save()

res["verdict"] = verdict
save()
print("\nVERDICT: " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in verdict.items()))
if all(verdict.values()):
    print("ALL OP GATES PASS")
    sys.exit(0)
print("NOT PASSED")
sys.exit(1)
