"""Hand test 3: tile size x batch size for gemm_q4_kernel_rdna3. G7 only.

Runs in the throwaway container, never in n02-vllm or Itko's sandbox.

Builds (256 and 128 reuse test2's cached builds; 64 is new):
  b256  BLOCK_KN_SIZE = THREADS_X = 256   control
  b128  BLOCK_KN_SIZE = THREADS_X = 128
  b64   BLOCK_KN_SIZE = THREADS_X = 64
Batches M = 1, 2, 4, 8 (M = 2..8 use the M_COUNT 2/4/8 kernels, which the tile
constant also changes; M >= 16 goes to the WMMA path and is out of scope).

Gate: every build at every M vs the installed op, judged against the installed
op's own run-to-run noise, plus a broken control that must fail. A build that
fails at some M is not timed at that M. If the control fails anywhere, stop.

Timing: graphed, distinct weight copies above 400 MB (shared by all kernels for a
shape, so every kernel reads the same data), median of 20 replays per run,
3 runs per kernel in shuffled order, reported as median and spread.
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
OUT = f"{HAND}/test3_results.json"
TARGET = 401 * 10**6
REPLAYS = 20
REPEATS = 3
MS = [1, 2, 4, 8]
BUILDS = {"b256": 256, "b128": 128, "b64": 64}
RMS_FLOOR, RMS_CAP, BROKEN_MIN = 0.01, 0.02, 0.10
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

# ---------------------------------------------------------------- build (identical to test2, so cached builds are reused)
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
    print(f"[{now()}] building {tag} (tile {bkn}); cached builds return at once, a new one takes about a minute", flush=True)
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
mods = {t: build(t, b) for t, b in BUILDS.items()}

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
    return (torch.randint(0, 2**31, (K // 8, N), device=dev, dtype=torch.int32),
            torch.randint(0, 2**31, (K // g, N // 8), device=dev, dtype=torch.int32),
            torch.randn(K // g, N, device=dev, dtype=torch.bfloat16) * 0.01,
            torch.empty(0, device=dev, dtype=torch.int))

def rms_rel(x, ref):
    d = x.float() - ref.float()
    return (d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()

installed = ops.gptq_gemm_rdna3
# The clone's entry point takes 5 arguments (no g_idx); the installed op takes 6.
def wrap(m):
    return lambda a, q, z, s, g, f: m.gemm(a, q, z, s, f)
kernels = {"installed": installed}
kernels.update({t: wrap(mods[t]) for t in BUILDS})
TAGS = list(kernels)

# ---------------------------------------------------------------- proof of which kernel runs
print("\n=== which GEMM kernel each entry point launches, at M = 1 and M = 8", flush=True)
res["kernels"] = {}
w = weights(5120, 2176)
for M in (1, 8):
    a = torch.randn(M, 5120, device=dev, dtype=torch.bfloat16)
    for t in TAGS:
        want = ("vllm" if t == "installed" else f"vllmht_{t}") + "::gptq_rdna3::gemm_q4_kernel_rdna3"
        try:
            kernels[t](a, *w, False); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                kernels[t](a, *w, False); torch.cuda.synchronize()
        except Exception as exc:
            fail(f"kernel check {t} M{M}", exc)
        names = [ev.name for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
        gemm = [nm for nm in names if "gemm" in nm.lower()]
        ok = len(gemm) == 1 and want in gemm[0]
        print(f"M{M} {t:9s} {'OK ' if ok else 'BAD'} {gemm[0][:100] if gemm else '(no gemm kernel seen)'}", flush=True)
        res["kernels"][f"{t}_M{M}"] = {"ok": ok, "names": names}
        if not ok and t != "installed":
            save()
            sys.exit(f"STOP: {t} did not launch its own kernel at M{M}")
        if not ok:
            print(f"   note: installed uses a different kernel at M{M}; its timing there compares against that kernel", flush=True)
    del a
del w
save()

# ---------------------------------------------------------------- gate
print(f"\n=== gate over {len(SHAPES)} shapes x M {MS}  (pass: RMS <= max(4 x self-noise, {RMS_FLOOR}), <= {RMS_CAP}; broken >= {BROKEN_MIN})", flush=True)
builds = list(BUILDS)
ok_at = {t: {M: True for M in MS} for t in builds}
summary = {t: {M: {"worst": 0.0, "limit": 0.0, "op": "", "min_broken": 9.9} for M in MS} for t in builds}
res["gate"] = {}
for M in MS:
    for op, (K, N) in SHAPES:
        try:
            w = weights(K, N)
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            ref = installed(a, *w, False)
            self_noise = max(rms_rel(installed(a, *w, False), ref) for _ in range(3))
            limit = min(max(4 * self_noise, RMS_FLOOR), RMS_CAP)
            wrong = (w[0], torch.randint(0, 2**31, w[1].shape, device=dev, dtype=torch.int32), w[2], w[3])
            row = {"self": self_noise, "limit": limit}
            for t in builds:
                e = rms_rel(kernels[t](a, *w, False), ref)
                b = rms_rel(kernels[t](a, *wrong, False), ref)
                row[t], row[t + "_broken"] = e, b
                sm = summary[t][M]
                if e / limit > sm["worst"] / max(sm["limit"], 1e-12):
                    sm.update(worst=e, limit=limit, op=op)
                sm["min_broken"] = min(sm["min_broken"], b)
                if e > limit or b < BROKEN_MIN:
                    ok_at[t][M] = False
            torch.cuda.synchronize()
        except Exception as exc:
            fail(f"gate M{M} {op}", exc)
        res["gate"][f"M{M}_{op}"] = row
        del w, a, ref, wrong
    torch.cuda.empty_cache()
print(f"{'build':6s} {'M':>2s} {'worst RMS':>10s} {'its limit':>10s} {'at':>9s} {'min broken':>11s}  verdict")
for t in builds:
    for M in MS:
        sm = summary[t][M]
        print(f"{t:6s} {M:2d} {sm['worst']:10.5f} {sm['limit']:10.5f} {sm['op']:>9s} {sm['min_broken']:11.3f}  "
              f"{'PASS' if ok_at[t][M] else 'FAIL'}", flush=True)
res["gate_passed"] = {t: {str(M): ok_at[t][M] for M in MS} for t in builds}
save()
if not all(ok_at["b256"].values()):
    print("STOP: the control build fails the gate, so the build itself is wrong.")
    sys.exit(1)

# ---------------------------------------------------------------- timing
def time_on(fn, a, ws):
    n = len(ws)
    def body():
        for w in ws:
            fn(a, w[0], w[1], w[2], w[3], False)
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

rng = random.Random(1234)
print(f"\n[{now()}] === timing: us per call, median of {REPEATS} shuffled runs (spread = (max-min)/median)", flush=True)
res["timing"] = {}
for op, (K, N) in SHAPES:
    try:
        n = max(100, -(-TARGET // wbytes(K, N)))
        ws = [weights(K, N) for _ in range(n)]
    except Exception as exc:
        fail(f"allocating {op}", exc)
    print(f"[{now()}] {op} (K{K} N{N}, {n} copies)", flush=True)
    for M in MS:
        tags = ["installed"] + [t for t in builds if ok_at[t][M]]
        runs = {t: [] for t in tags}
        try:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            for _ in range(REPEATS):
                order = tags[:]
                rng.shuffle(order)
                for t in order:
                    runs[t].append(time_on(kernels[t], a, ws))
        except Exception as exc:
            fail(f"timing {op} M{M}", exc)
        med = {t: statistics.median(runs[t]) for t in tags}
        spread = {t: (max(runs[t]) - min(runs[t])) / med[t] for t in tags}
        res["timing"][f"{op}_M{M}"] = {"median_us": med, "spread": spread, "runs": runs}
        cells = "  ".join(f"{t} {med[t]:6.2f} ({spread[t] * 100:3.0f}%)" for t in tags)
        rel = f"b256/inst {med['b256'] / med['installed']:.3f}  " + "  ".join(
            f"{t}/b256 {med[t] / med['b256']:.3f}" for t in tags if t.startswith("b") and t != "b256")
        print(f"   M{M}: {cells}   |  {rel}", flush=True)
        del a
    del ws
    torch.cuda.empty_cache()
    save()

# ---------------------------------------------------------------- per-token summary
print(f"\n[{now()}] === 9 TP8 linears per decode step, ms (one step decodes M sequences)")
print(f"{'M':>2s} " + " ".join(f"{t:>10s}" for t in TAGS) + f"   {'b256/inst':>10s} " + " ".join(f"{t + '/b256':>10s}" for t in builds if t != "b256"))
res["per_step_ms"] = {}
for M in MS:
    tot = {}
    for t in TAGS:
        vals = [res["timing"].get(f"{op}_M{M}", {}).get("median_us", {}).get(t) for op in TP8]
        tot[t] = sum(v * CALLS[op] / 1000.0 for v, op in zip(vals, TP8)) if all(v is not None for v in vals) else None
    res["per_step_ms"][str(M)] = tot
    cells = " ".join(f"{tot[t]:10.3f}" if tot[t] is not None else f"{'not timed':>10s}" for t in TAGS)
    rel = f"{tot['b256'] / tot['installed']:10.3f} " + " ".join(
        f"{tot[t] / tot['b256']:10.3f}" if tot[t] is not None else f"{'-':>10s}" for t in builds if t != "b256")
    print(f"{M:2d} {cells}   {rel}")
worst_spread = max(v for r in res["timing"].values() for v in r["spread"].values())
print(f"largest run-to-run spread anywhere: {worst_spread * 100:.0f}%")
res["worst_spread"] = worst_spread
save()
print(f"\n[{now()}] saved -> {OUT}")
