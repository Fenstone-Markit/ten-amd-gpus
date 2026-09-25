"""Hand test 2: tile-size variant of gemm_q4_kernel_rdna3. G7 only.

Runs in a throwaway container (see run command), never in n02-vllm or Itko's sandbox.

Builds the clone's kernel twice:
  b256  BLOCK_KN_SIZE = THREADS_X = 256   control, must match the installed op
  b128  BLOCK_KN_SIZE = THREADS_X = 128   variant
Each build gets its own C++ namespace and function names, so it cannot silently
bind to the installed vLLM kernel. The profiler then proves which kernel ran.

Gate (before any timing): each build vs the installed op on identical inputs,
judged against the installed op's own run-to-run noise (atomics are
non-deterministic), plus a broken control (wrong zeros) that must fail.

Timing: graphed, distinct weight copies above 400 MB, median of 20 replays,
installed / b256 / b128 interleaved per shape.
"""
import os, sys, re, json, time, shutil, statistics
os.environ["ROCR_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HIP_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_PLUGINS"] = ""
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"

import torch
from torch.utils.cpp_extension import load, ROCM_HOME
from torch.profiler import profile, ProfilerActivity
from vllm import _custom_ops as ops

dev = "cuda"
HAND = "/workspace/percard-work/hand"
SRC = f"{HAND}/src"
OUT = f"{HAND}/test2_results.json"
TARGET = 401 * 10**6
REPLAYS = 20
RMS_FLOOR = 0.01    # a different atomic order in bf16 is expected to land well below this
RMS_CAP = 0.02      # never pass anything worse than this
BROKEN_MIN = 0.10   # the broken control must be at least this far off
res = {}

def now():
    return time.strftime("%H:%M:%S")

def save():
    json.dump(res, open(OUT, "w"), indent=2)

def fail(where, exc):
    print(f"\nFAILED in {where}: {type(exc).__name__}: {exc}", flush=True)
    res["failed"] = {"where": where, "error": f"{type(exc).__name__}: {exc}"}
    save()
    sys.exit(1)

# ---------------------------------------------------------------- preflight (fails fast, before any build)
if torch.cuda.device_count() != 1:
    sys.exit(f"ABORT: {torch.cuda.device_count()} cards visible, expected 1")
free = torch.cuda.mem_get_info()[0] / 2**30
print(f"[{now()}] visible cards: 1, free VRAM: {free:.1f} GiB", flush=True)
if free < 20:
    sys.exit("ABORT: card is not empty, refusing to run")
ninja = shutil.which("ninja")
hipcc = os.path.join(ROCM_HOME or "", "bin", "hipcc")
print(f"ninja: {ninja}\nROCM_HOME: {ROCM_HOME}\nhipcc: {hipcc} ({'found' if os.path.exists(hipcc) else 'MISSING'})", flush=True)
if not ninja:
    sys.exit("ABORT: ninja not found; torch cannot build extensions without it")
if not os.path.exists(hipcc):
    sys.exit("ABORT: hipcc not found under ROCM_HOME")
try:
    base = open(f"{SRC}/q_gemm_rdna3.cu").read()
    hdr = open(f"{SRC}/qdq_4_rdna3.cuh").read()
except Exception as exc:
    fail("reading sources (host prep step not done?)", exc)

# ---------------------------------------------------------------- build
def stub(tag):
    return f'''

// ---- appended by test2.py: standalone build glue ----
// The WMMA path is only taken for bf16 with M >= 16; decode (M = 1) never reaches it.
torch::Tensor ht_wmma_{tag}(torch::Tensor a, torch::Tensor b_q_weight,
                            torch::Tensor b_qzeros,
                            torch::Tensor b_scales, bool use_v2_format) {{
  TORCH_CHECK(false, "wmma path is not built in this test (M must be < 16)");
  return a;
}}
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("gemm", &ht_gemm_{tag}); }}
'''

def rename(text, tag):
    # Unique namespace and entry points per build, so nothing can bind to the installed vLLM symbols.
    text = re.sub(r"\bvllm\b", f"vllmht_{tag}", text)
    text = re.sub(r"\bgptq_gemm_rdna3_wmma\b", f"ht_wmma_{tag}", text)
    text = re.sub(r"\bgptq_gemm_rdna3\b", f"ht_gemm_{tag}", text)
    return text

def build(tag, bkn):
    d = f"{HAND}/build/{tag}"
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, "lock")
    if os.path.exists(lock):   # left behind by an interrupted build; load() would wait on it forever
        os.remove(lock)
        print(f"removed stale build lock in {d}", flush=True)
    s = base
    for name in ("BLOCK_KN_SIZE", "THREADS_X"):
        s, n = re.subn(rf"^#define {name} 256\s*$", f"#define {name} {bkn}", s, flags=re.M)
        if n != 1:
            fail(f"build {tag}", RuntimeError(f"expected one '#define {name} 256', found {n}"))
    open(f"{d}/q_gemm_rdna3.cu", "w").write(rename(s, tag) + stub(tag))
    open(f"{d}/qdq_4_rdna3.cuh", "w").write(rename(hdr, tag))
    print(f"[{now()}] building {tag} (BLOCK_KN_SIZE = THREADS_X = {bkn}), quiet for a few minutes...", flush=True)
    t0 = time.time()
    try:
        mod = load(name=f"gq4_{tag}", sources=[f"{d}/q_gemm_rdna3.cu"],
                   build_directory=d, extra_include_paths=[d],
                   extra_cuda_cflags=["-O3", "-DUSE_ROCM"], verbose=False)
    except Exception as exc:
        fail(f"build {tag}", exc)
    print(f"[{now()}] built {tag} in {time.time() - t0:.0f} s", flush=True)
    return mod

print("\n=== build (first run compiles; later runs reuse the build)", flush=True)
mods = {"b256": build("b256", 256), "b128": build("b128", 128)}

# ---------------------------------------------------------------- helpers
TP8 = {"q": (5120, 1536), "k": (5120, 256), "v": (5120, 256), "o": (768, 5120),
       "gate": (5120, 2176), "up": (5120, 2176), "down": (2176, 5120),
       "qkvz": (5120, 2048), "gdn_out": (768, 5120)}
CALLS = {"q": 16, "k": 16, "v": 16, "o": 16, "gate": 64, "up": 64,
         "down": 64, "qkvz": 48, "gdn_out": 47}
SHAPES = list(TP8.items()) + [("gate_tp2", (5120, 8704))]

def wbytes(K, N, g=32):
    return (K // 8) * N * 4 + (K // g) * N * 2 + (K // g) * (N // 8) * 4

def weights(K, N, g=32):
    # call order: qweight, qzeros, scales, g_idx
    return (torch.randint(0, 2**31, (K // 8, N), device=dev, dtype=torch.int32),
            torch.randint(0, 2**31, (K // g, N // 8), device=dev, dtype=torch.int32),
            torch.randn(K // g, N, device=dev, dtype=torch.bfloat16) * 0.01,
            torch.empty(0, device=dev, dtype=torch.int))

def rms_rel(x, ref):
    d = x.float() - ref.float()
    return (d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()

installed = ops.gptq_gemm_rdna3
# The clone's entry point takes 5 arguments (no g_idx); the installed op takes 6.
# g_idx is empty for this model, so dropping it computes the same thing.
kernels = {"installed": installed,
           "b256": lambda a, q, z, s, g, f: mods["b256"].gemm(a, q, z, s, f),
           "b128": lambda a, q, z, s, g, f: mods["b128"].gemm(a, q, z, s, f)}

# ---------------------------------------------------------------- proof of which kernel runs
print("\n=== which GEMM kernel each entry point actually launches", flush=True)
res["kernels"] = {}
expect = {"installed": "vllm::gptq_rdna3::gemm_q4_kernel_rdna3",
          "b256": "vllmht_b256::gptq_rdna3::gemm_q4_kernel_rdna3",
          "b128": "vllmht_b128::gptq_rdna3::gemm_q4_kernel_rdna3"}
w = weights(5120, 2176)
a = torch.randn(1, 5120, device=dev, dtype=torch.bfloat16)
for t, fn in kernels.items():
    try:
        fn(a, *w, False); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn(a, *w, False); torch.cuda.synchronize()
    except Exception as exc:
        fail(f"kernel check {t}", exc)
    names = [ev.name for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
    gemm = [nm for nm in names if "gemm_q4_kernel_rdna3" in nm]
    ok = len(gemm) == 1 and expect[t] in gemm[0]
    print(f"{t:9s} {'OK ' if ok else 'BAD'} {gemm[0][:110] if gemm else '(no gemm kernel seen)'}", flush=True)
    res["kernels"][t] = {"names": names, "ok": ok}
    if not ok:
        save()
        sys.exit(f"STOP: {t} did not launch its own kernel ({expect[t]}); results would be meaningless")
del w, a
save()

# ---------------------------------------------------------------- gate
print(f"\n=== gate: RMS error relative to RMS of the installed op's output", flush=True)
print(f"pass: <= max(4 x installed self-noise, {RMS_FLOOR}) and <= {RMS_CAP}; broken control must be >= {BROKEN_MIN}")
print(f"{'op':8s} {'self':>9s} {'b256':>9s} {'b128':>9s} {'b256 brk':>9s} {'b128 brk':>9s}")
res["gate"] = {}
passed = {"b256": True, "b128": True}
for op, (K, N) in SHAPES:
    try:
        w = weights(K, N)
        a = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        ref = installed(a, *w, False)
        self_noise = max(rms_rel(installed(a, *w, False), ref) for _ in range(3))
        limit = min(max(4 * self_noise, RMS_FLOOR), RMS_CAP)
        wrong = (w[0], torch.randint(0, 2**31, w[1].shape, device=dev, dtype=torch.int32), w[2], w[3])
        row = {"self": self_noise, "limit": limit}
        for t in ("b256", "b128"):
            row[t] = rms_rel(kernels[t](a, *w, False), ref)
            row[t + "_broken"] = rms_rel(kernels[t](a, *wrong, False), ref)
            if row[t] > limit or row[t + "_broken"] < BROKEN_MIN:
                passed[t] = False
        torch.cuda.synchronize()
    except Exception as exc:
        fail(f"gate {op}", exc)
    print(f"{op:8s} {row['self']:9.5f} {row['b256']:9.5f} {row['b128']:9.5f} "
          f"{row['b256_broken']:9.3f} {row['b128_broken']:9.3f}", flush=True)
    res["gate"][op] = row
    del w, a, ref, wrong
torch.cuda.empty_cache()
res["gate_passed"] = passed
save()
print(f"gate result: b256 {'PASS' if passed['b256'] else 'FAIL'}, b128 {'PASS' if passed['b128'] else 'FAIL'}")
if not passed["b256"]:
    print("STOP: the control build does not reproduce the installed op, so the build itself is wrong.")
    sys.exit(1)

# ---------------------------------------------------------------- timing
def graph_time(body, n_calls):
    for _ in range(3):
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(REPLAYS):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        t.append(s.elapsed_time(e) * 1000.0 / n_calls)
    del g
    return statistics.median(t)

def time_kernel(fn, K, N):
    n = max(100, -(-TARGET // wbytes(K, N)))
    a = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
    ws = [weights(K, N) for _ in range(n)]
    def body():
        for w in ws:
            fn(a, w[0], w[1], w[2], w[3], False)
    us = graph_time(body, n)
    del ws, a
    torch.cuda.empty_cache()
    return us

tags = ["installed", "b256"] + (["b128"] if passed["b128"] else [])
if not passed["b128"]:
    print("\nb128 FAILED the gate: it will not be timed.")
print(f"\n[{now()}] === timing, us per call (graphed, distinct copies)", flush=True)
print(f"{'op':8s} {'calls':>5s} " + " ".join(f"{t:>10s}" for t in tags) + f" {'b256/inst':>10s}"
      + (f" {'b128/inst':>10s}" if "b128" in tags else ""))
res["timing"] = {}
totals = {t: 0.0 for t in tags}
control_off = []
for op, (K, N) in SHAPES:
    row = {}
    for t in tags:
        try:
            row[t] = time_kernel(kernels[t], K, N)
        except Exception as exc:
            fail(f"timing {op} {t}", exc)
    ratio = row["b256"] / row["installed"]
    if not 0.95 <= ratio <= 1.05:
        control_off.append(op)
    line = f"{op:8s} {CALLS.get(op, 0):5d} " + " ".join(f"{row[t]:10.2f}" for t in tags) + f" {ratio:10.3f}"
    if "b128" in tags:
        line += f" {row['b128'] / row['installed']:10.3f}"
    print(line, flush=True)
    if op in CALLS:
        for t in tags:
            totals[t] += row[t] * CALLS[op] / 1000.0
    res["timing"][op] = row
    save()
print("TP8 9-op total, ms per token: " + ", ".join(f"{t} {totals[t]:.3f}" for t in tags))
res["tp8_total_ms"] = totals
res["control_off_by_more_than_5pct"] = control_off
if control_off:
    print(f"WARNING: control b256 differs from installed by more than 5% on {control_off}. "
          "The build differs from production (flags or source), so b128's gain must be read against b256, not installed.")
else:
    print("control b256 reproduces installed within 5% on every shape.")
save()
print(f"\n[{now()}] saved -> {OUT}")
