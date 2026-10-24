# vmm-repro: does live GPU memory get overwritten when a card fills up, with PyTorch expandable segments?
# Order mirrors a GLM worker: four marked 512 KiB blocks first (small pool), then "weights", then rounds of
# activation-like allocations filled with random bf16 up to a headroom of H MiB free, freed again
# (MODE ec: empty_cache on odd rounds; none: never; oom: never, and allocate past H until PyTorch runs out). Args: H_MiB ROUNDS MODE
import sys, time, torch
H = int(sys.argv[1]); R = int(sys.argv[2]); MODE = sys.argv[3]; dev = "cuda"; MiB = 2**20
print(f"devices {torch.cuda.device_count()}, {torch.cuda.get_device_name(0)}, alloc conf: {torch.cuda.memory._get_current_allocator() if hasattr(torch.cuda.memory, '_get_current_allocator') else ''}", flush=True)
assert torch.cuda.device_count() == 1, "must see exactly one GPU"
torch.manual_seed(1)
can = []
for i in range(4):
    t = torch.arange(131072, dtype=torch.int32, device=dev) * 7 + 0x5A5A0000 + i
    can.append([t, t.cpu().numpy().copy()])
torch.cuda.synchronize()
first_hit = [None]
def check(tag):
    n_hit = 0
    for i, c in enumerate(can):
        g = c[0].cpu().numpy(); d = g != c[1]; n = int(d.sum())
        if n:
            n_hit += 1; f = int(d.argmax())
            print(f"HIT {tag} block {i}: {n} of 131072 words changed, first bytes {g[f:f+4].tobytes().hex()}", flush=True)
            c[1] = g.copy()
            if first_hit[0] is None: first_hit[0] = tag
    return n_hit
free = lambda: torch.cuda.mem_get_info()[0] // MiB
weights = []
while free() > H + 2048:
    w = torch.empty(64 * MiB, dtype=torch.float32, device=dev); w.fill_(1.0); weights.append(w)
torch.cuda.synchronize()
print(f"weights {len(weights) * 256} MiB; free {free()} MiB; headroom target {H} MiB; canary at {hex(can[0][0].data_ptr())}", flush=True)
check("after-weights")
sizes = [578, 64, 20, 6, 1, 0.25]
t0 = time.time(); total_hits = 0; ooms = 0
for r in range(R):
    acts = []; k = 0
    while free() > H or MODE == "oom":
        n = int(sizes[k % len(sizes)] * MiB) // 2; k += 1
        try:
            a = torch.empty(n, dtype=torch.bfloat16, device=dev); a.normal_(); acts.append(a)
        except torch.OutOfMemoryError:
            ooms += 1; break
    torch.cuda.synchronize(); total_hits += check(f"round {r} full ({len(acts)} buffers, free {free()} MiB)")
    del acts
    if MODE == "ec" and r % 2: torch.cuda.empty_cache()
    torch.cuda.synchronize(); total_hits += check(f"round {r} freed")
print(f"RESULT mode {MODE}, headroom {H} MiB, {R} rounds in {time.time() - t0:.0f} s: {total_hits} block hits, first at: {first_hit[0]}, OOM retries {ooms}", flush=True)
