"""Hand test 4: how much of the remaining per-call cost is the output-zeroing fill. G7 only.

Runs in the throwaway container, never in n02-vllm or Itko's sandbox.

Builds (b128 and b64 reuse cached builds; the two nf builds are new):
  b128, b64       as in test3 (output zeroed by torch::zeros, then atomically accumulated)
  b128nf, b64nf   TIMING PROBES ONLY: torch::zeros replaced by torch::empty, so the fill
                  kernel is gone and the output is garbage. They are never gated and
                  must never be used for anything but this timing comparison.
Also timed: the fill on its own (torch.zeros of the output shape, graphed).

Timing: graphed, distinct weight copies above 400 MB shared by all kernels for a shape,
median of 20 replays per run, 5 runs per kernel in shuffled order, M = 1 and 4.
"""
import os, sys, re, json, time, shutil, random, statistics
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
OUT = f"{HAND}/test4_results.json"
TARGET = 401 * 10**6
REPLAYS = 20
REPEATS = 5
MS = [1, 4]
BUILDS = {"b128": (128, False), "b64": (64, False), "b128nf": (128, True), "b64nf": (64, True)}
ZERO_LINE = "at::Tensor c = torch::zeros({size_m, size_n}, opts);"
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
print(f"[{now()}] visible cards: 1, free VRAM: {free:.1f} GiB", flush=True)
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
if base.count(ZERO_LINE) != 1:
    sys.exit(f"ABORT: expected exactly one output-zeroing line, found {base.count(ZERO_LINE)}")

# ---------------------------------------------------------------- build (glue identical to test2/test3, so b128 and b64 are reused)
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

def build(tag, bkn, nofill):
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
    if nofill:
        s = s.replace(ZERO_LINE, "at::Tensor c = torch::empty({size_m, size_n}, opts);  // TIMING PROBE: output invalid")
    open(f"{d}/q_gemm_rdna3.cu", "w").write(rename(s, tag) + stub(tag))
    open(f"{d}/qdq_4_rdna3.cuh", "w").write(rename(hdr, tag))
    print(f"[{now()}] building {tag} (tile {bkn}{', NO FILL' if nofill else ''}); cached builds return at once, a new one takes about a minute", flush=True)
    t0 = time.time()
    try:
        mod = load(name=f"gq4_{tag}", sources=[f"{d}/q_gemm_rdna3.cu"],
                   build_directory=d, extra_include_paths=[d],
                   extra_cuda_cflags=["-O3", "-DUSE_ROCM"], verbose=False)
    except Exception as exc:
        fail(f"build {tag}", exc)
    print(f"[{now()}] built {tag} in {time.time() - t0:.0f} s", flush=True)
    return mod

print("\n=== build", flush=True)
mods = {t: build(t, b, nf) for t, (b, nf) in BUILDS.items()}

# ---------------------------------------------------------------- helpers
TP8 = {"q": (5120, 1536), "k": (5120, 256), "v": (5120, 256), "o": (768, 5120),
       "gate": (5120, 2176), "up": (5120, 2176), "down": (2176, 5120),
       "qkvz": (5120, 2048), "gdn_out": (768, 5120)}
CALLS = {"q": 16, "k": 16, "v": 16, "o": 16, "gate": 64, "up": 64,
         "down": 64, "qkvz": 48, "gdn_out": 47}

def wbytes(K, N, g=32):
    return (K // 8) * N * 4 + (K // g) * N * 2 + (K // g) * (N // 8) * 4

def weights(K, N, g=32):
    return (torch.randint(0, 2**31, (K // 8, N), device=dev, dtype=torch.int32),
            torch.randint(0, 2**31, (K // g, N // 8), device=dev, dtype=torch.int32),
            torch.randn(K // g, N, device=dev, dtype=torch.bfloat16) * 0.01,
            torch.empty(0, device=dev, dtype=torch.int))

def wrap(m):
    return lambda a, q, z, s, g, f: m.gemm(a, q, z, s, f)

kernels = {"installed": ops.gptq_gemm_rdna3}
kernels.update({t: wrap(mods[t]) for t in BUILDS})
TAGS = list(kernels) + ["fill"]

# ---------------------------------------------------------------- proof: own kernel, and the probes really have no fill
print("\n=== device kernels per call at M = 1 (zeroed builds: fill + gemm; probes: gemm only)", flush=True)
res["kernels"] = {}
w = weights(5120, 2176)
a = torch.randn(1, 5120, device=dev, dtype=torch.bfloat16)
for t in kernels:
    want = ("vllm" if t == "installed" else f"vllmht_{t}") + "::gptq_rdna3::gemm_q4_kernel_rdna3"
    try:
        kernels[t](a, *w, False); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            kernels[t](a, *w, False); torch.cuda.synchronize()
    except Exception as exc:
        fail(f"kernel check {t}", exc)
    names = [ev.name for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
    gemm = [nm for nm in names if "gemm" in nm.lower()]
    fills = [nm for nm in names if "Fill" in nm]
    nofill = t.endswith("nf")
    ok = len(gemm) == 1 and want in gemm[0] and (len(fills) == 0 if nofill else len(fills) == 1)
    print(f"{t:9s} {'OK ' if ok else 'BAD'} kernels {len(names)}, fill {len(fills)}  {gemm[0][:80] if gemm else '(no gemm)'}", flush=True)
    res["kernels"][t] = {"ok": ok, "names": names}
    if not ok:
        save()
        sys.exit(f"STOP: {t} does not launch what it should")
del w, a
save()

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

def time_tag(t, a, ws, M, N):
    n = len(ws)
    if t == "fill":
        def body():
            for _ in range(n):
                torch.zeros((M, N), device=dev, dtype=torch.bfloat16)
    else:
        fn = kernels[t]
        def body():
            for w in ws:
                fn(a, w[0], w[1], w[2], w[3], False)
    return time_body(body, n)

rng = random.Random(4321)
print(f"\n[{now()}] === timing: us per call, median of {REPEATS} shuffled runs (spread = (max-min)/median)", flush=True)
res["timing"] = {}
for op, (K, N) in TP8.items():
    try:
        n = max(100, -(-TARGET // wbytes(K, N)))
        ws = [weights(K, N) for _ in range(n)]
    except Exception as exc:
        fail(f"allocating {op}", exc)
    print(f"[{now()}] {op} (K{K} N{N}, {n} copies)", flush=True)
    for M in MS:
        runs = {t: [] for t in TAGS}
        try:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            for _ in range(REPEATS):
                order = TAGS[:]
                rng.shuffle(order)
                for t in order:
                    runs[t].append(time_tag(t, a, ws, M, N))
        except Exception as exc:
            fail(f"timing {op} M{M}", exc)
        med = {t: statistics.median(runs[t]) for t in TAGS}
        spread = {t: (max(runs[t]) - min(runs[t])) / med[t] for t in TAGS}
        res["timing"][f"{op}_M{M}"] = {"median_us": med, "spread": spread, "runs": runs}
        cells = "  ".join(f"{t} {med[t]:6.2f} ({spread[t] * 100:3.0f}%)" for t in TAGS)
        print(f"   M{M}: {cells}", flush=True)
        print(f"        fill share: b128 {med['b128'] - med['b128nf']:5.2f} us, b64 {med['b64'] - med['b64nf']:5.2f} us, fill alone {med['fill']:5.2f} us", flush=True)
        del a
    del ws
    torch.cuda.empty_cache()
    save()

# ---------------------------------------------------------------- per-step summary
print(f"\n[{now()}] === 9 TP8 linears per decode step, ms")
res["per_step_ms"] = {}
for M in MS:
    def tot(pick):
        return sum(pick(res["timing"][f"{op}_M{M}"]["median_us"]) * CALLS[op] / 1000.0 for op in TP8)
    row = {t: tot(lambda m, t=t: m[t]) for t in kernels}
    row["best_tile"] = tot(lambda m: min(m["b128"], m["b64"]))
    row["best_tile_nofill"] = tot(lambda m: min(m["b128nf"], m["b64nf"]))
    row["fill_alone"] = tot(lambda m: m["fill"])
    res["per_step_ms"][str(M)] = row
    print(f"M{M}: " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
worst = max(v for r in res["timing"].values() for v in r["spread"].values())
worst_at = max(((k, t) for k, r in res["timing"].items() for t, v in r["spread"].items()),
               key=lambda kt: res["timing"][kt[0]]["spread"][kt[1]])
print(f"largest run-to-run spread: {worst * 100:.0f}% at {worst_at[0]} {worst_at[1]}")
res["worst_spread"] = {"value": worst, "at": list(worst_at)}
save()
print(f"\n[{now()}] saved -> {OUT}")
