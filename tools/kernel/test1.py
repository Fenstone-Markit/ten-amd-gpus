"""Hand test 1 + 2: instrument check and gptq_gemm_rdna3 floor map. G7 only.

Run from the host:
  podman exec n02-sandbox python /workspace/percard-work/hand/test1.py

1a  known answer: graphed 25 MB device copy must land near 771 GB/s
1b  comparator: read-only kernel (sum) at the same byte counts as the GEMMs
1c  cache control: same GEMM with distinct weight copies vs one shared copy,
    plus reproduction of Itko's k (23.62 us) and gate (28.38 us)
1d  kernels launched by one gptq_gemm_rdna3 call (names only, no durations)
2   floor map: K x N sweep, distinct copies
"""
import os, sys, json, statistics
os.environ["ROCR_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HIP_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-d580007fa41ec467"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_PLUGINS"] = ""

import torch
from vllm import _custom_ops as ops

dev = "cuda"
torch.manual_seed(0)
TARGET = 401 * 10**6   # per-graph bytes above the 96 MB Infinity Cache
REPLAYS = 20
OUT = "/workspace/percard-work/hand/test1_results.json"
res = {}

# Safety: exactly one visible card, and it must be empty (a brain card has ~1.2 GB free)
if torch.cuda.device_count() != 1:
    sys.exit(f"ABORT: {torch.cuda.device_count()} cards visible, expected 1")
free = torch.cuda.mem_get_info()[0] / 2**30
print(f"visible cards: 1, free VRAM: {free:.1f} GiB")
if free < 20:
    sys.exit("ABORT: card is not empty, refusing to run")

def save():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

def fail(where, exc):
    # Stop, do not continue: a GPU fault can leave the context unusable,
    # and every later number would be garbage.
    print(f"\nFAILED in {where}: {type(exc).__name__}: {exc}", flush=True)
    res["failed"] = {"where": where, "error": f"{type(exc).__name__}: {exc}"}
    save()
    sys.exit(1)

def wbytes(K, N, g=32):
    return (K // 8) * N * 4 + (K // g) * N * 2 + (K // g) * (N // 8) * 4

def weights(K, N, g=32):
    # returned in call order: qweight, qzeros, scales, g_idx
    return (torch.randint(0, 2**31, (K // 8, N), device=dev, dtype=torch.int32),
            torch.randint(0, 2**31, (K // g, N // 8), device=dev, dtype=torch.int32),
            torch.randn(K // g, N, device=dev, dtype=torch.bfloat16) * 0.01,
            torch.empty(0, device=dev, dtype=torch.int))

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
    return statistics.median(t), min(t), max(t)

def gemm_us(K, N, shared=False):
    per = wbytes(K, N)
    n = max(100, -(-TARGET // per))
    a = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
    ws = [weights(K, N)] * n if shared else [weights(K, N) for _ in range(n)]
    def body():
        for w in ws:
            ops.gptq_gemm_rdna3(a, w[0], w[1], w[2], w[3], False)
    med, lo, hi = graph_time(body, n)
    del ws, a
    torch.cuda.empty_cache()
    return med, lo, hi, n, per

# 1a known answer
print("\n=== 1a  graphed device copy, 25 MiB (known answer: near 771 GB/s, read+write)")
size = 25 * 2**20
n = -(-TARGET // size)
src = [torch.empty(size, dtype=torch.uint8, device=dev) for _ in range(n)]
dst = [torch.empty(size, dtype=torch.uint8, device=dev) for _ in range(n)]
def body():
    for s_, d_ in zip(src, dst):
        d_.copy_(s_)
try:
    med, lo, hi = graph_time(body, n)
except Exception as exc:
    fail("1a copy", exc)
gbps = 2 * size / (med * 1e-6) / 1e9
print(f"copy 25 MiB x{n}: {med:.2f} us/copy (min {lo:.2f} max {hi:.2f}), {gbps:.1f} GB/s")
res["1a_copy"] = {"us": med, "gbps": gbps}
save()
del src, dst; torch.cuda.empty_cache()

# 1b read-only comparator at GEMM byte counts
print("\n=== 1b  graphed read-only sum at GEMM byte counts (what reading N bytes costs)")
res["1b_sum"] = {}
for mb in (0.76, 2.27, 6.44, 25.8):
    nb = int(mb * 10**6) // 2 * 2
    k = max(100, -(-TARGET // nb))
    xs = [torch.empty(nb // 2, dtype=torch.bfloat16, device=dev).normal_() for _ in range(k)]
    def body():
        for x in xs:
            x.sum()
    try:
        med, lo, hi = graph_time(body, k)
    except Exception as exc:
        fail(f"1b sum {mb} MB", exc)
    gbps = nb / (med * 1e-6) / 1e9
    print(f"sum {mb:5.2f} MB x{k}: {med:7.2f} us/call  {gbps:6.1f} GB/s")
    res["1b_sum"][str(mb)] = {"us": med, "gbps": gbps}
    del xs; torch.cuda.empty_cache()
save()

# 1c cache control and reproduction
print("\n=== 1c  gptq_gemm_rdna3: distinct copies (VRAM) vs one shared copy (cache)")
res["1c_gemm"] = {}
for label, K, N, itko in (("k_tp8", 5120, 256, 23.62), ("gate_tp8", 5120, 2176, 28.38),
                          ("gate_tp2", 5120, 8704, 38.95)):
    try:
        d = gemm_us(K, N, shared=False)
        s = gemm_us(K, N, shared=True)
    except Exception as exc:
        fail(f"1c {label}", exc)
    diff = (d[0] - itko) / itko * 100
    print(f"{label:9s} K{K} N{N} {d[4]/1e6:5.2f} MB | distinct {d[0]:7.2f} us "
          f"(Itko {itko}, {diff:+.1f}%) | shared {s[0]:7.2f} us | distinct/shared {d[0]/s[0]:.2f}")
    res["1c_gemm"][label] = {"distinct_us": d[0], "shared_us": s[0], "itko_us": itko,
                             "mb": d[4] / 1e6}
save()

# 1d kernels per call
print("\n=== 1d  kernels launched by one call (names only; profiler durations are not trusted)")
from torch.profiler import profile, ProfilerActivity
w = weights(5120, 2176)
a = torch.randn(1, 5120, device=dev, dtype=torch.bfloat16)
for _ in range(3):
    ops.gptq_gemm_rdna3(a, w[0], w[1], w[2], w[3], False)
torch.cuda.synchronize()
try:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        ops.gptq_gemm_rdna3(a, w[0], w[1], w[2], w[3], False)
        torch.cuda.synchronize()
except Exception as exc:
    fail("1d profiler", exc)
names = [ev.name for ev in prof.events()
         if ev.device_type == torch.autograd.DeviceType.CUDA]
print(f"{len(names)} device events (0 means the profiler saw no device activity, not zero kernels):")
for nm in names:
    print("  ", nm[:140])
res["1d_kernels"] = names
save()
del w, a; torch.cuda.empty_cache()

# 2 floor map
print("\n=== 2  floor map, distinct copies: us per call (MB, GB/s)")
res["2_sweep"] = {}
Ns = (256, 1024, 2176, 5120, 8704)
print("K \\ N  " + "".join(f"{n:>24d}" for n in Ns))
for K in (768, 2048, 5120):
    row = f"{K:5d}  "
    for N in Ns:
        try:
            med, lo, hi, n, per = gemm_us(K, N)
        except Exception as exc:
            fail(f"2 sweep K{K} N{N}", exc)
        gb = per / (med * 1e-6) / 1e9
        row += f"{med:9.2f} ({per/1e6:5.2f}, {gb:5.0f})"
        res["2_sweep"][f"{K}x{N}"] = {"us": med, "mb": per / 1e6, "gbps": gb}
    print(row, flush=True)
    save()

save()
print(f"\nsaved -> {OUT}")
