"""rdna_ar_test8.py: the RDNA3 all-reduce (vLLM PR #57767, plus the TP=8 butterfly) on 2, 4 or 8 cards, against RCCL.

Same lifecycle and graph-only use as rdna_ar_test.py (which passed on G7 and G9). Per size (1 to 24 tokens at
hidden size 5120):
  1. exact sum: rank r contributes r + 1; every element must equal world * (world + 1) / 2. Any miss: STOP.
  2. random data: every rank's output must be bit-identical to rank 0's (STOP otherwise), and its RMS error
     against the float64 sum must be at most 1.25 x RCCL's own error on the same data (STOP otherwise).
     Above 2 ranks the summation order differs from RCCL's, so bit-for-bit equality with RCCL is not expected.
  3. timing: 50 calls in one graph, 12 replays, slowest rank, events outside the graph; RCCL the same way.
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

if world not in (2, 4, 8):
    stop(f"this test runs on 2, 4 or 8 ranks, got {world}")
TARGET = world * (world + 1) // 2
if any(k.startswith("VLLM_RDNA") for k in os.environ):
    stop("VLLM_RDNA* tuning variables are set; the kernel's defaults must be measured")

torch.ops.load_library(args.library)
ops = torch.ops._rdna_custom_ar
arch = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]
log(f"rdna_ar_test8: {world} ranks, {torch.cuda.get_device_name(local)} ({arch}); library {args.library}")

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
log(f"setup complete on all {world} ranks: rank 0 peers {peers}, shared buffers opened and registered")

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
log(f"\n{'tokens':>6s} {'KB':>7s}   exact  ranks agree  error (RDNA / RCCL)   {'RDNA us':>8s} {'RCCL us':>8s}  RDNA/RCCL")
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
    bad = int(slowest(int((out != TARGET).sum().item())))
    if bad:
        teardown()
        stop(f"{tokens} tokens: {bad} elements are not {TARGET} after the sum; the kernel is not adding correctly here")
    # 2. random data against RCCL, through a replay
    gen = torch.Generator(device="cpu").manual_seed(9000 + tokens * 10 + rank)
    inp.copy_(torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)); out.zero_()
    torch.cuda.synchronize(); barrier()
    g1.replay(); torch.cuda.synchronize()
    ref = inp.clone(); dist.all_reduce(ref); torch.cuda.synchronize()
    gathered_in = [torch.empty_like(inp) for _ in range(world)]; dist.all_gather(gathered_in, inp)
    exact = sum(x.double() for x in gathered_in); scale = exact.pow(2).mean().sqrt()
    err_rdna = slowest(float(((out.double() - exact).pow(2).mean().sqrt() / scale).item()))
    err_rccl = slowest(float(((ref.double() - exact).pow(2).mean().sqrt() / scale).item()))
    gathered_out = [torch.empty_like(out) for _ in range(world)]; dist.all_gather(gathered_out, out)
    disagree = int(slowest(sum(int((g != gathered_out[0]).sum().item()) for g in gathered_out)))
    mism = int(slowest(int((out != ref).sum().item())))
    worst = err_rdna / err_rccl if err_rccl > 0 else 0.0
    if disagree:
        teardown()
        stop(f"{tokens} tokens: {disagree} elements differ between ranks; every rank must hold the same sum")
    if err_rdna > 1.25 * err_rccl:
        teardown()
        stop(f"{tokens} tokens: RMS error {err_rdna:.5f} against the exact sum, above 1.25 x RCCL's {err_rccl:.5f}")
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
    rows.append({"tokens": tokens, "bytes": nbytes, "exact": True, "ranks_agree": True, "mismatch_vs_rccl": mism,
                 "rms_err_rdna": err_rdna, "rms_err_rccl": err_rccl, "rdna_us": us_rdna, "rccl_us": us_rccl})
    log(f"{tokens:6d} {nbytes / 1e3:7.1f}   yes    yes          {err_rdna:.5f} / {err_rccl:.5f}     "
        f"{us_rdna:8.1f} {us_rccl:8.1f}  {us_rdna / us_rccl:8.2f}")

teardown()
if rank == 0:
    os.makedirs(args.out, exist_ok=True)
    json.dump({"world": world, "arch": arch, "hidden": HIDDEN, "stride": STRIDE, "rows": rows,
               "note": f"{world} ranks, graph replays"},
              open(os.path.join(args.out, f"rdna_ar_test_w{world}.json"), "w"), indent=2)
    log(f"\nsaved -> {os.path.join(args.out, f'rdna_ar_test_w{world}.json')}")
dist.destroy_process_group()
