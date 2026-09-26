"""allreduce_bench.py: all-reduce cost on the brain's 8 cards, at decode and prefill message sizes.

Run with torchrun, 8 processes, one card each, once per algorithm (NCCL_ALGO is read when the
communicator is created, so each algorithm needs its own run):
  NCCL_ALGO=Tree torchrun --nproc_per_node 8 allreduce_bench.py --label tree --out /out
  NCCL_ALGO=Ring torchrun --nproc_per_node 8 allreduce_bench.py --label ring --out /out
                 torchrun --nproc_per_node 8 allreduce_bench.py --label auto --out /out

Message sizes are the brain's own: hidden size 5120 in bf16 = 10,240 bytes per token. Decode sends
1 to 4 tokens per all-reduce; prefill sends up to 2,048 (21 MB).

Method, per size:
  1. correctness: rank r fills its tensor with r + 1; after the sum every element must be exactly
     1 + 2 + ... + world (36 for 8 ranks). Any mismatch stops the run.
  2. timing (GPU): 50 all-reduces captured in one CUDA graph, replayed 12 times, CUDA events around
     each replay. Decode runs its all-reduces inside a CUDA graph, so this is how the brain pays them.
     Each replay's time is the SLOWEST rank's (an all-reduce is only as fast as its slowest card).
     Reported: median over replays, per call.
  3. timing (CPU, --backend gloo, for testing this script only): eager loop, perf_counter.
Bus bandwidth is reported the standard way: bytes x 2 x (world - 1) / world / time.
"""
import argparse, json, os, statistics, sys, time
import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--out", required=True)
p.add_argument("--backend", default="nccl", choices=("nccl", "gloo"))
p.add_argument("--tokens", default="1,2,4,8,32,128,512,1024,2048")
p.add_argument("--calls", type=int, default=50)
p.add_argument("--replays", type=int, default=12)
args = p.parse_args()

HIDDEN = 5120
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", rank))
gpu = args.backend == "nccl"
if gpu:
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
else:
    dev = torch.device("cpu")
from datetime import timedelta
if gpu:
    dist.init_process_group(args.backend, timeout=timedelta(seconds=120), device_id=dev)
else:
    dist.init_process_group(args.backend, timeout=timedelta(seconds=120))

def barrier():
    if gpu:
        dist.barrier(device_ids=[local])
    else:
        dist.barrier()
dtype = torch.bfloat16

def log(msg):
    if rank == 0:
        print(msg, flush=True)

def slowest(x):
    t = torch.tensor([x], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

log(f"allreduce_bench: label {args.label}, backend {args.backend}, world {world}, "
    f"NCCL_ALGO={os.environ.get('NCCL_ALGO', '(unset: library chooses)')}")
if gpu:
    log(f"cards: {[torch.cuda.get_device_name(local)]} x {world}")
want = world * (world + 1) // 2
rows = []
mode_note = None
for tokens in [int(t) for t in args.tokens.split(",")]:
    nbytes = tokens * HIDDEN * 2
    x = torch.full((tokens, HIDDEN), float(rank + 1), dtype=dtype, device=dev)
    dist.all_reduce(x)
    if gpu:
        torch.cuda.synchronize()
    bad = int((x != want).sum().item())
    bad = int(slowest(bad))
    if bad:
        log(f"STOP: {tokens} tokens: {bad} elements are not {want} after the sum. The all-reduce is wrong.")
        dist.destroy_process_group(); sys.exit(2)

    buf = torch.zeros((tokens, HIDDEN), dtype=dtype, device=dev)   # zeros stay zero: repeated sums cannot overflow
    times = []
    mode = "cpu-eager"
    if gpu:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                dist.all_reduce(buf)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # Graphed, as decode pays it. If this build cannot capture collectives, fall back to eager
        # timing and say so: eager includes launch overhead, so it overstates decode's cost.
        g, capture_error = None, None
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(args.calls):
                    dist.all_reduce(buf)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
        except Exception as exc:
            g, capture_error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
            torch.cuda.synchronize()
        ok_everywhere = slowest(0.0 if g is not None else 1.0) == 0.0   # every rank must use the same mode
        mode = "graphed" if ok_everywhere else "EAGER"
        if not ok_everywhere and mode_note is None:
            mode_note = capture_error or "another rank failed to capture"
            log(f"  NOTE: CUDA-graph capture of all-reduce failed ({mode_note}). Timing EAGER instead: "
                f"includes launch overhead, so decode's real cost is lower than shown.")
        barrier()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.replays):
            barrier(); torch.cuda.synchronize()
            ev0.record()
            if mode == "graphed":
                g.replay()
            else:
                for _ in range(args.calls):
                    dist.all_reduce(buf)
            ev1.record(); torch.cuda.synchronize()
            times.append(slowest(ev0.elapsed_time(ev1) * 1000.0 / args.calls))
        del g
    else:
        for _ in range(3):
            dist.all_reduce(buf)
        for _ in range(args.replays):
            barrier()
            t0 = time.perf_counter()
            for _ in range(args.calls):
                dist.all_reduce(buf)
            times.append(slowest((time.perf_counter() - t0) * 1e6 / args.calls))
    us = statistics.median(times)
    busbw = nbytes * 2 * (world - 1) / world / (us * 1e-6) / 1e9
    rows.append({"tokens": tokens, "bytes": nbytes, "us_per_call": us, "busbw_GBps": busbw,
                 "replays_us": times, "correct": True, "mode": mode})
    log(f"  {tokens:5d} tokens {nbytes / 1e3:9.1f} KB   {us:9.1f} us per all-reduce   bus bandwidth {busbw:6.2f} GB/s   [{mode}]")

if rank == 0:
    dec = next((r for r in rows if r["tokens"] == 1), None)
    if dec:
        log(f"decode, 1 token: {dec['us_per_call']:.1f} us per all-reduce x 128 per token = "
            f"{dec['us_per_call'] * 128 / 1000:.2f} ms per token [{dec['mode']}]")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"allreduce_{args.label}.json")
    json.dump({"label": args.label, "backend": args.backend, "world": world,
               "nccl_algo": os.environ.get("NCCL_ALGO"), "calls_per_replay": args.calls, "rows": rows,
               "capture_note": mode_note},
              open(path, "w"), indent=2)
    log(f"saved -> {path}")
barrier()
dist.destroy_process_group()
