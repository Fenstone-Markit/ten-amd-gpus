# vmm-repro-04-2card: the fixed-pressure test (vmm-repro-03) on two cards in one process, rounds alternating between them,
# so a pool that mixed cards would show up as hits or errors. Args: TARGET_MiB ROUNDS
import sys, torch; MiB=2**20; T=int(sys.argv[1]); R=int(sys.argv[2])
assert torch.cuda.device_count()==2, "must see exactly two GPUs"
can={}; w={}
for d in (0,1):
    can[d]=[]
    for i in range(4):
        t=torch.arange(131072,dtype=torch.int32,device=f"cuda:{d}")*7+0x5A5A0000+i+16*d; can[d].append([t,t.cpu().numpy().copy()])
def check(tag):
    n=0
    for d in (0,1):
        for i,c in enumerate(can[d]):
            g=c[0].cpu().numpy(); x=g!=c[1]
            if x.any(): n+=1; print(f"HIT {tag} card {d} block {i}: {int(x.sum())} words",flush=True); c[1]=g.copy()
    return n
for d in (0,1):
    w[d]=[]; torch.cuda.set_device(d)
    while torch.cuda.mem_get_info(d)[0]//MiB>150+2048: x=torch.empty(64*MiB,dtype=torch.float32,device=f"cuda:{d}"); x.fill_(1.0); w[d].append(x)
torch.cuda.synchronize(0); torch.cuda.synchronize(1); check("after-weights")
sizes=[578,64,20,6,1,0.25]; hits=0; oom=0
for r in range(R):
    d=r%2 if False else (r//2)%2; a=[]; k=0; got=0.0
    while got<T:
        s=sizes[k%6]; k+=1
        try: b=torch.empty(int(s*MiB)//2,dtype=torch.bfloat16,device=f"cuda:{d}"); b.normal_(); a.append(b); got+=s
        except torch.OutOfMemoryError: oom+=1; break
    torch.cuda.synchronize(d); hits+=check(f"round {r} card {d} full")
    print(f"round {r} card {d}: {len(a)} buffers, free {torch.cuda.mem_get_info(d)[0]//MiB} MiB",flush=True)
    del a
    if r%2: torch.cuda.empty_cache()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1); hits+=check(f"round {r} freed")
print(f"RESULT two cards, fixed {T} MiB per round, {R} rounds: {hits} block hits, OOM {oom}",flush=True)
