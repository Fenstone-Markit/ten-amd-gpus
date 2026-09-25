"""Hand test 5: the GEMM production actually runs. G7 only.

Runs in the throwaway container, never in n02-vllm or Itko's sandbox.
Check first that this image's triton_w4a16.py is byte-identical to production's.

Production's compiled graph runs triton_w4a16_gemm_kernel for all 255 quantised
linears per token, with the large-batch config frozen in (BLOCK 128 x 32, BLOCK_K
clamped to the group size 32). Shapes here are production's merged ones.

Kernels timed:
  triton_prod   triton_w4a16_gemm_kernel launched directly, BLOCK 128/32/32 (as compiled)
  triton_eager  the Python wrapper, which picks by actual M (32/32/32 at M = 1)
  hip_inst      installed gptq_gemm_rdna3 (HIP), which production rejected
  hip_b128, hip_b64   the tile variants gated in test3

Gate: triton_prod must reproduce triton_eager's output (same math, different tile),
and a broken control (wrong zeros) must fail. HIP builds were gated in test3.
"""
import os, sys, re, json, time, shutil, random, statistics
os.environ["ROCR_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HIP_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_PLUGINS"] = ""
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"

import torch
import triton
from torch.utils.cpp_extension import load, ROCM_HOME
from torch.profiler import profile, ProfilerActivity
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm, triton_w4a16_gemm_kernel)

dev = "cuda"
HAND = "/workspace/percard-work/hand"
SRC = f"{HAND}/src"
OUT = f"{HAND}/test5_results.json"
G = 32
TARGET = 401 * 10**6
REPLAYS = 20
REPEATS = 3
MS = [1, 4]
RMS_OK, BROKEN_MIN = 0.005, 0.10
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

# ---------------------------------------------------------------- preflight
if torch.cuda.device_count() != 1:
    sys.exit(f"ABORT: {torch.cuda.device_count()} cards visible, expected 1")
free = torch.cuda.mem_get_info()[0] / 2**30
print(f"[{now()}] visible cards: 1, free VRAM: {free:.1f} GiB, triton {triton.__version__}", flush=True)
if free < 20:
    sys.exit("ABORT: card is not empty, refusing to run")
hipcc = os.path.join(ROCM_HOME or "", "bin", "hipcc")
if not shutil.which("ninja") or not os.path.exists(hipcc):
    sys.exit(f"ABORT: ninja {shutil.which('ninja')} / hipcc {hipcc} missing")
try:
    base = open(f"{SRC}/q_gemm_rdna3.cu").read()
    hdr = open(f"{SRC}/qdq_4_rdna3.cuh").read()
except Exception as exc:
    fail("reading sources", exc)

# ---------------------------------------------------------------- HIP builds (glue identical to test3, so cached builds are reused)
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
    text = re.sub(r"\bvllm\b", f"vllmht_{tag}", text)
    text = re.sub(r"\bgptq_gemm_rdna3_wmma\b", f"ht_wmma_{tag}", text)
    text = re.sub(r"\bgptq_gemm_rdna3\b", f"ht_gemm_{tag}", text)
    return text

def build(tag, bkn):
    d = f"{HAND}/build/{tag}"
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, "lock")
    if os.path.exists(lock):
        os.remove(lock)
        print(f"removed stale build lock in {d}", flush=True)
    s = base
    for name in ("BLOCK_KN_SIZE", "THREADS_X"):
        s, n = re.subn(rf"^#define {name} 256\s*$", f"#define {name} {bkn}", s, flags=re.M)
        if n != 1:
            fail(f"build {tag}", RuntimeError(f"expected one '#define {name} 256', found {n}"))
    open(f"{d}/q_gemm_rdna3.cu", "w").write(rename(s, tag) + stub(tag))
    open(f"{d}/qdq_4_rdna3.cuh", "w").write(rename(hdr, tag))
    print(f"[{now()}] building {tag} (tile {bkn}); cached builds return at once", flush=True)
    t0 = time.time()
    try:
        mod = load(name=f"gq4_{tag}", sources=[f"{d}/q_gemm_rdna3.cu"],
                   build_directory=d, extra_include_paths=[d],
                   extra_cuda_cflags=["-O3", "-DUSE_ROCM"], verbose=False)
    except Exception as exc:
        fail(f"build {tag}", exc)
    print(f"[{now()}] built {tag} in {time.time() - t0:.0f} s", flush=True)
    return mod

print("\n=== HIP builds", flush=True)
mods = {"b128": build("b128", 128), "b64": build("b64", 64)}

# ---------------------------------------------------------------- shapes, weights, kernels
# Production's merged shapes, per TP8 rank: (K, N), calls per token. 255 calls in total.
SHAPES = {"qkv": ((5120, 2048), 16), "o": ((768, 5120), 16), "gate_up": ((5120, 4352), 64),
          "down": ((2176, 5120), 64), "qkvz": ((5120, 2048), 48), "gdn_out": ((768, 5120), 47)}
assert sum(c for _, c in SHAPES.values()) == 255

def wbytes(K, N):
    return K * N // 2 + (K // G) * N * 2 + (K // G) * (N // 8) * 4

def hip_weights(K, N):
    # HIP layout, call order: qweight [K/8, N], qzeros [K/G, N/8], scales [K/G, N], g_idx
    return (torch.randint(0, 2**31, (K // 8, N), device=dev, dtype=torch.int32),
            torch.randint(0, 2**31, (K // G, N // 8), device=dev, dtype=torch.int32),
            torch.randn(K // G, N, device=dev, dtype=torch.bfloat16) * 0.01,
            torch.empty(0, device=dev, dtype=torch.int))

def tri_weights(K, N):
    # Triton layout: b_q [K, N/8], scales [K/G, N], qzeros [K/G, N/8]
    return (torch.randint(0, 2**31, (K, N // 8), device=dev, dtype=torch.int32),
            torch.randn(K // G, N, device=dev, dtype=torch.bfloat16) * 0.01,
            torch.randint(0, 2**31, (K // G, N // 8), device=dev, dtype=torch.int32))

def triton_prod(a, w):
    b_q, sc, qz = w
    M, K = a.shape
    N = b_q.shape[1] * 8
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 32))
    triton_w4a16_gemm_kernel[grid](a, b_q, sc, qz, c, M, N, K,
                                   a.stride(0), a.stride(1), b_q.stride(0), b_q.stride(1),
                                   c.stride(0), c.stride(1), group_size=G,
                                   HAS_ZP=True, ZP_BIAS=8, BLOCK_M=128, BLOCK_N=32, BLOCK_K=32)
    return c

def triton_eager(a, w):
    b_q, sc, qz = w
    return triton_w4a16_gemm(a, b_q, sc, qz, G)

def hip_wrap(m):
    return lambda a, w: m.gemm(a, w[0], w[1], w[2], False)

KERNELS = {"triton_prod": (triton_prod, "tri"), "triton_eager": (triton_eager, "tri"),
           "hip_inst": (lambda a, w: ops.gptq_gemm_rdna3(a, w[0], w[1], w[2], w[3], False), "hip"),
           "hip_b128": (hip_wrap(mods["b128"]), "hip"), "hip_b64": (hip_wrap(mods["b64"]), "hip")}
TAGS = list(KERNELS)

def rms_rel(x, ref):
    d = x.float() - ref.float()
    return (d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()

# ---------------------------------------------------------------- kernel proof (also compiles the Triton kernels)
print(f"\n[{now()}] === which kernels each entry point launches, M = 1 (first Triton call compiles, a few seconds)", flush=True)
res["kernels"] = {}
K, N = SHAPES["gate_up"][0]
a = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
wt, wh = tri_weights(K, N), hip_weights(K, N)
want = {"triton_prod": "triton_w4a16_gemm_kernel", "triton_eager": "triton_w4a16_gemm_kernel",
        "hip_inst": "vllm::gptq_rdna3::gemm_q4_kernel_rdna3",
        "hip_b128": "vllmht_b128::gptq_rdna3::gemm_q4_kernel_rdna3",
        "hip_b64": "vllmht_b64::gptq_rdna3::gemm_q4_kernel_rdna3"}
for t, (fn, lay) in KERNELS.items():
    w = wt if lay == "tri" else wh
    try:
        fn(a, w); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn(a, w); torch.cuda.synchronize()
    except Exception as exc:
        fail(f"kernel check {t}", exc)
    names = [ev.name for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
    gemm = [nm for nm in names if "gemm" in nm.lower()]
    ok = len(gemm) == 1 and want[t] in gemm[0]
    print(f"{t:12s} {'OK ' if ok else 'BAD'} kernels {len(names)}  {gemm[0][:90] if gemm else '(no gemm)'}", flush=True)
    res["kernels"][t] = {"ok": ok, "names": names}
    if not ok:
        save()
        sys.exit(f"STOP: {t} does not launch what it should")
save()

# ---------------------------------------------------------------- gate: our direct launch reproduces the wrapper
print(f"\n=== gate: triton_prod vs triton_eager (pass RMS <= {RMS_OK}; broken control must be >= {BROKEN_MIN})", flush=True)
res["gate"] = {}
gate_ok = True
for op, ((K, N), _) in SHAPES.items():
    for M in MS:
        try:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = tri_weights(K, N)
            ref = triton_eager(a, w)
            e = rms_rel(triton_prod(a, w), ref)
            wrong = (w[0], w[1], torch.randint(0, 2**31, w[2].shape, device=dev, dtype=torch.int32))
            b = rms_rel(triton_prod(a, wrong), ref)
            torch.cuda.synchronize()
        except Exception as exc:
            fail(f"gate {op} M{M}", exc)
        ok = e <= RMS_OK and b >= BROKEN_MIN
        gate_ok &= ok
        res["gate"][f"{op}_M{M}"] = {"rms": e, "broken": b, "ok": ok}
        print(f"{op:8s} M{M}  rms {e:.6f}  broken {b:.3f}  {'PASS' if ok else 'FAIL'}", flush=True)
save()
if not gate_ok:
    sys.exit("STOP: the direct Triton launch does not reproduce the wrapper; timing it would be meaningless")

# ---------------------------------------------------------------- timing
def time_body(body, n):
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
        t.append(s.elapsed_time(e) * 1000.0 / n)
    del g
    return statistics.median(t)

rng = random.Random(5555)
print(f"\n[{now()}] === timing: us per call, median of {REPEATS} shuffled runs (spread = (max-min)/median)", flush=True)
res["timing"] = {}
for op, ((K, N), calls) in SHAPES.items():
    n = max(100, -(-TARGET // wbytes(K, N)))
    try:
        W = {"tri": [tri_weights(K, N) for _ in range(n)], "hip": [hip_weights(K, N) for _ in range(n)]}
    except Exception as exc:
        fail(f"allocating {op}", exc)
    print(f"[{now()}] {op} (K{K} N{N}, {calls} calls/token, {n} copies per layout)", flush=True)
    for M in MS:
        runs = {t: [] for t in TAGS}
        try:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            for _ in range(REPEATS):
                order = TAGS[:]
                rng.shuffle(order)
                for t in order:
                    fn, lay = KERNELS[t]
                    ws = W[lay]
                    def body(fn=fn, ws=ws):
                        for w in ws:
                            fn(a, w)
                    runs[t].append(time_body(body, n))
        except Exception as exc:
            fail(f"timing {op} M{M}", exc)
        med = {t: statistics.median(runs[t]) for t in TAGS}
        spread = {t: (max(runs[t]) - min(runs[t])) / med[t] for t in TAGS}
        res["timing"][f"{op}_M{M}"] = {"median_us": med, "spread": spread, "runs": runs}
        print(f"   M{M}: " + "  ".join(f"{t} {med[t]:6.2f} ({spread[t] * 100:3.0f}%)" for t in TAGS), flush=True)
        del a
    del W
    torch.cuda.empty_cache()
    save()

# ---------------------------------------------------------------- per-token summary
print(f"\n[{now()}] === 255 quantised linears per decode step (TP8 rank), ms")
res["per_step_ms"] = {}
for M in MS:
    tot = {t: sum(res["timing"][f"{op}_M{M}"]["median_us"][t] * c for op, ((_, _), c) in SHAPES.items()) / 1000.0
           for t in TAGS}
    tot["hip_best_tile"] = sum(min(res["timing"][f"{op}_M{M}"]["median_us"][t] for t in ("hip_b128", "hip_b64")) * c
                               for op, ((_, _), c) in SHAPES.items()) / 1000.0
    res["per_step_ms"][str(M)] = tot
    print(f"M{M}: " + "  ".join(f"{k} {v:.3f}" for k, v in tot.items()))
    p = tot["triton_prod"]
    print(f"     relative to production (triton_prod): " + "  ".join(f"{k} {v / p:.3f}" for k, v in tot.items() if k != "triton_prod"))
worst = max(v for r in res["timing"].values() for v in r["spread"].values())
print(f"largest run-to-run spread: {worst * 100:.0f}%")
res["worst_spread"] = worst
save()
print(f"\n[{now()}] saved -> {OUT}")
