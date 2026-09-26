"""qr_test.py: can vLLM's QuickReduce run on RDNA3 over PCIe? 2 ranks: G7 and G9.

QuickReduce is AMD's MI300 all-reduce. This image compiles it for gfx1100, but vLLM only enables it
for gfx94/gfx95 with XGMI links. This test calls the compiled ops directly, bypassing those checks.

Uncompressed level (FP) only, no bf16-to-fp16 cast: the sum must be exact.
Per message size:
  1. exact sum: rank r contributes r + 1, so every element must be 1 + 2 = 3. Any miss: STOP.
  2. random data, bit for bit against RCCL. A 2-rank sum is one addition, and a + b gives identical
     bits in either order, so a correct QuickReduce must match RCCL exactly. Mismatches are counted;
     anything beyond one bf16 step: STOP.
  3. timing: 50 calls in one CUDA graph, 12 replays, slowest rank, for QuickReduce and for RCCL.
     If graph capture fails, eager timing, labelled.
2-rank results answer "does it run, and is it exact". They do not predict 8 ranks.

Run with torch.distributed.run, 2 processes, and a fence to G7 and G9 (see the run command).
"""
import argparse, json, os, statistics, sys
from datetime import timedelta
import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--tokens", default="1,2,4,8,32,128,512,1024,2048")
p.add_argument("--calls", type=int, default=50)
p.add_argument("--replays", type=int, default=12)
args = p.parse_args()

HIDDEN = 5120
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
dist.init_process_group("nccl", timeout=timedelta(seconds=60), device_id=dev)

def log(msg):
    if rank == 0:
        print(msg, flush=True)

def barrier():
    dist.barrier(device_ids=[local])

def slowest(x):
    t = torch.tensor([float(x)], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

def stop(msg, ptr=None):
    log(f"STOP: {msg}")
    dist.destroy_process_group()
    sys.exit(2)

if world != 2:
    stop(f"this test is for exactly 2 ranks, got {world}")

import vllm._custom_ops as ops
from vllm.distributed.device_communicators.quick_all_reduce import QuickReduceRegime
FP = QuickReduceRegime.FP.value
log(f"qr_test: 2 ranks, cards {[torch.cuda.get_device_name(local)]} x 2; QuickReduce levels "
    f"{[(r.name, r.value) for r in QuickReduceRegime]}; using FP = {FP} (uncompressed)")

ptr = ops.init_custom_qr(rank, world, None)
max_bytes = ops.qr_max_size()
handle = ops.qr_get_handle(ptr)
handles = [None] * world
dist.all_gather_object(handles, handle)
ops.qr_open_handles(ptr, handles)
torch.cuda.synchronize(); barrier()
log(f"QuickReduce initialised on both ranks; buffer handles exchanged and opened; max message {max_bytes / 2**20:.0f} MiB")

def qr(inp, out):
    ops.qr_all_reduce(ptr, inp, out, FP, False)

def timed(fn, label):
    """50 calls of fn in one graph, 12 replays, slowest rank per replay; eager fallback, labelled."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g, mode = None, "graphed"
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(args.calls):
                fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    except Exception as exc:
        g, mode = None, "EAGER"
        torch.cuda.synchronize()
        log(f"  NOTE: graph capture of {label} failed ({type(exc).__name__}: {str(exc)[:120]}); timing eager")
    if slowest(1.0 if g is None else 0.0) != 0.0:
        g, mode = None, "EAGER"      # every rank must time the same way
    barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(args.replays):
        barrier(); torch.cuda.synchronize()
        e0.record()
        if g is not None:
            g.replay()
        else:
            for _ in range(args.calls):
                fn()
        e1.record(); torch.cuda.synchronize()
        t.append(slowest(e0.elapsed_time(e1) * 1000.0 / args.calls))
    del g
    return statistics.median(t), mode

rows = []
log(f"\n{'tokens':>6s} {'KB':>9s}   exact  vs RCCL (bit-for-bit)   {'QR us':>8s} {'RCCL us':>8s}  QR/RCCL")
for tokens in [int(t) for t in args.tokens.split(",")]:
    nbytes = tokens * HIDDEN * 2
    if nbytes > max_bytes:
        log(f"{tokens:6d} skipped: {nbytes} bytes is above QuickReduce's max")
        continue
    # 1. exact sum
    x = torch.full((tokens, HIDDEN), float(rank + 1), dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(x)
    qr(x, out); torch.cuda.synchronize()
    bad = int(slowest(int((out != 3).sum().item())))
    if bad:
        stop(f"{tokens} tokens: {bad} elements are not 3 after the sum; QuickReduce is not adding correctly here")
    # 2. random data, bit for bit against RCCL
    gen = torch.Generator(device="cpu").manual_seed(7000 + tokens * 10 + rank)
    r = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16).to(dev)
    out_qr = torch.empty_like(r)
    qr(r, out_qr)
    ref = r.clone(); dist.all_reduce(ref)
    torch.cuda.synchronize()
    diff = (out_qr.float() - ref.float()).abs()
    step = torch.ldexp(torch.ones_like(ref.float()), torch.frexp(ref.float().abs().clamp_min(1e-30))[1] - 8)
    mism = int(slowest(int((out_qr != ref).sum().item())))
    worst_steps = slowest(float((diff / step).max().item()))
    if worst_steps > 1.0:
        stop(f"{tokens} tokens: QuickReduce differs from RCCL by up to {worst_steps:.1f} bf16 steps; not a correct sum")
    # 3. timing
    inp_t = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)   # zeros: repeated sums cannot overflow
    out_t = torch.empty_like(inp_t)
    us_qr, m_qr = timed(lambda: qr(inp_t, out_t), "QuickReduce")
    buf = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    us_rc, m_rc = timed(lambda: dist.all_reduce(buf), "RCCL")
    rows.append({"tokens": tokens, "bytes": nbytes, "exact": True, "mismatch_vs_rccl": mism,
                 "worst_steps_vs_rccl": worst_steps, "qr_us": us_qr, "qr_mode": m_qr, "rccl_us": us_rc, "rccl_mode": m_rc})
    log(f"{tokens:6d} {nbytes / 1e3:9.1f}   yes    {'identical' if mism == 0 else f'{mism} differ (<= 1 step)':22s} "
        f"{us_qr:8.1f} {us_rc:8.1f}  {us_qr / us_rc:6.2f}   [{m_qr}/{m_rc}]")

ops.qr_destroy(ptr)
if rank == 0:
    os.makedirs(args.out, exist_ok=True)
    json.dump({"world": world, "level": "FP", "max_bytes": max_bytes, "rows": rows,
               "note": "2 ranks (G7, G9): answers whether it runs and is exact; does not predict 8 ranks"},
              open(os.path.join(args.out, "qr_test.json"), "w"), indent=2)
    log(f"\nsaved -> {os.path.join(args.out, 'qr_test.json')}")
barrier()
dist.destroy_process_group()
