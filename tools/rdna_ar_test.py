"""rdna_ar_test.py: the RDNA3 all-reduce from vLLM PR #57767, on 2 cards (G7, G9), against RCCL.

Loads the library built from csrc/rocm/rdna_custom_all_reduce.cu and follows the kernel's own lifecycle:
  allocate_shared_buffer_and_handle (before any capture) -> exchange handles -> open the peer's
  -> init_custom_ar -> register_buffer -> all_reduce -> dispose, close peer handle, free own buffer.

The wrapper in vLLM uses this kernel only inside CUDA graphs, so every check runs through a graph replay:
static input and output buffers, capture once, then refill the input and replay.
Sizes: 1, 2, 4, 8, 16, 24 tokens at hidden size 5120 (the brain's), inside the wrapper's RDNA3 range
(numel < 128K). Per size:
  1. exact sum: rank r contributes r + 1; every element must be 3. Any miss: STOP.
  2. random data against RCCL, bit for bit (a 2-rank sum is one addition); beyond one bf16 step: STOP.
  3. timing: 50 calls in one graph, 12 replays, slowest rank, events outside the graph; RCCL the same way.
If the kernel cannot be captured in a graph: STOP (it is graph-only by design).
2 ranks answer "does it work here, and how fast at 2 ranks". They do not predict 8 ranks.
"""
import argparse, json, os, statistics, sys
from datetime import timedelta
import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--library", required=True)
p.add_argument("--out", required=True)
p.add_argument("--tokens", default="1,2,4,8,16,24")
p.add_argument("--calls", type=int, default=50)
p.add_argument("--replays", type=int, default=12)
args = p.parse_args()

HIDDEN = 5120
STRIDE = 128 * 8192 * 2          # the wrapper's buffer stride, 2 MiB
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

def stop(msg):
    log(f"STOP: {msg}")
    dist.destroy_process_group()
    sys.exit(2)

if world != 2:
    stop(f"this test is for exactly 2 ranks, got {world}")
if any(k.startswith("VLLM_RDNA") for k in os.environ):
    stop("VLLM_RDNA* tuning variables are set; the kernel's defaults must be measured")

torch.ops.load_library(args.library)
ops = torch.ops._rdna_custom_ar
arch = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]
log(f"rdna_ar_test: 2 ranks, {torch.cuda.get_device_name(local)} ({arch}); library {args.library}")

# ---- setup, exactly per the kernel's lifecycle, all before any graph capture
peers = list(ops.get_required_peer_ranks(rank, world))
own_ptr, handle = ops.allocate_shared_buffer_and_handle(STRIDE, world)
rank_data = torch.empty(ops.rank_data_size(), dtype=torch.uint8, device=dev)
handles = [None] * world
dist.all_gather_object(handles, handle)
pointers = [0] * world
pointers[rank] = own_ptr
opened = []
for peer in peers:
    q = ops.open_mem_handle(handles[peer])
    opened.append(q)
    pointers[peer] = q
meta = ops.meta_size()
payloads = [q + meta if q else 0 for q in pointers]
ctx = ops.init_custom_ar(pointers, rank_data, rank, STRIDE)
ops.register_buffer(ctx, payloads)
payload = payloads[rank]
torch.cuda.synchronize(); barrier()
log(f"setup complete on both ranks: peers {peers}, shared buffers opened and registered")

def teardown():
    torch.cuda.synchronize(); barrier()
    ops.dispose(ctx)
    for q in opened:
        ops.close_mem_handle(q)
    barrier()                      # peers unmap before the owner frees
    ops.free_shared_buffer(own_ptr)

def capture(fn, calls, warm):
    # The RDNA kernel is graph-only in vLLM and never called eagerly; RCCL gets a normal warm-up.
    if warm:
        fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for _ in range(calls):
                fn()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"
    return g, None

def time_graph(g):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize(); barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(args.replays):
        barrier(); torch.cuda.synchronize()
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        t.append(slowest(e0.elapsed_time(e1) * 1000.0 / args.calls))
    return statistics.median(t)

rows = []
log(f"\n{'tokens':>6s} {'KB':>7s}   exact  vs RCCL (bit-for-bit)    {'RDNA us':>8s} {'RCCL us':>8s}  RDNA/RCCL")
for tokens in [int(x) for x in args.tokens.split(",")]:
    nbytes = tokens * HIDDEN * 2
    inp = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    out = torch.zeros_like(inp)
    rdna = lambda: ops.all_reduce(ctx, inp, out, payload, STRIDE)
    g1, err = capture(rdna, 1, warm=False)
    if g1 is None:
        teardown()
        stop(f"{tokens} tokens: the kernel could not be captured in a CUDA graph ({err})")
    # 1. exact sum, through a replay
    inp.fill_(float(rank + 1)); out.zero_(); torch.cuda.synchronize(); barrier()
    g1.replay(); torch.cuda.synchronize()
    bad = int(slowest(int((out != 3).sum().item())))
    if bad:
        teardown()
        stop(f"{tokens} tokens: {bad} elements are not 3 after the sum; the kernel is not adding correctly here")
    # 2. random data against RCCL, through a replay
    gen = torch.Generator(device="cpu").manual_seed(9000 + tokens * 10 + rank)
    inp.copy_(torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)); out.zero_()
    torch.cuda.synchronize(); barrier()
    g1.replay(); torch.cuda.synchronize()
    ref = inp.clone(); dist.all_reduce(ref); torch.cuda.synchronize()
    step = torch.ldexp(torch.ones_like(ref.float()), torch.frexp(ref.float().abs().clamp_min(1e-30))[1] - 8)
    worst = slowest(float(((out.float() - ref.float()).abs() / step).max().item()))
    mism = int(slowest(int((out != ref).sum().item())))
    if worst > 1.0:
        teardown()
        stop(f"{tokens} tokens: differs from RCCL by up to {worst:.1f} bf16 steps; not a correct sum")
    del g1
    # 3. timing, 50 calls per graph, against RCCL the same way
    inp.zero_()
    g50, err = capture(rdna, args.calls, warm=False)
    if g50 is None:
        teardown()
        stop(f"{tokens} tokens: capturing {args.calls} calls failed ({err})")
    us_rdna = time_graph(g50); del g50
    buf = torch.zeros_like(inp)
    gr, err = capture(lambda: dist.all_reduce(buf), args.calls, warm=True)
    if gr is None:
        teardown()
        stop(f"{tokens} tokens: RCCL graph capture failed ({err})")
    us_rccl = time_graph(gr); del gr
    rows.append({"tokens": tokens, "bytes": nbytes, "exact": True, "mismatch_vs_rccl": mism,
                 "worst_steps_vs_rccl": worst, "rdna_us": us_rdna, "rccl_us": us_rccl})
    log(f"{tokens:6d} {nbytes / 1e3:7.1f}   yes    {'identical' if mism == 0 else f'{mism} differ (<= 1 step)':22s} "
        f"{us_rdna:8.1f} {us_rccl:8.1f}  {us_rdna / us_rccl:8.2f}")

teardown()
if rank == 0:
    os.makedirs(args.out, exist_ok=True)
    json.dump({"world": world, "arch": arch, "hidden": HIDDEN, "stride": STRIDE, "rows": rows,
               "note": "2 ranks (G7, G9), graph replays; does not predict 8 ranks"},
              open(os.path.join(args.out, "rdna_ar_test.json"), "w"), indent=2)
    log(f"\nsaved -> {os.path.join(args.out, 'rdna_ar_test.json')}")
dist.destroy_process_group()
