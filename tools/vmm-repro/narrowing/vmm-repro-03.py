# vmm-repro-03: like vmm-repro.py, but each round allocates a fixed byte total (TARGET MiB, the per-round fill seen on the control stack),
# so pressure does not depend on what mem_get_info reports. empty_cache after odd rounds. Prints every round. Args: TARGET_MiB ROUNDS
import sys, torch; MiB=2**20; dev="cuda"; T=int(sys.argv[1]); R=int(sys.argv[2])
print(torch.__version__, torch.version.hip, flush=True); assert torch.cuda.device_count()==1
can=[]
for i in range(4):
    t=torch.arange(131072,dtype=torch.int32,device=dev)*7+0x5A5A0000+i; can.append([t,t.cpu().numpy().copy()])
def check(tag):
    n=0
    for i,c in enumerate(can):
        g=c[0].cpu().numpy(); d=g!=c[1]
        if d.any(): n+=1; f=int(d.argmax()); print(f"HIT {tag} block {i}: {int(d.sum())} words, first bytes {g[f:f+4].tobytes().hex()}",flush=True); c[1]=g.copy()
    return n
free=lambda: torch.cuda.mem_get_info()[0]//MiB
w=[]
while free()>150+2048: x=torch.empty(64*MiB,dtype=torch.float32,device=dev); x.fill_(1.0); w.append(x)
torch.cuda.synchronize(); print(f"weights {len(w)*256} MiB, free {free()} MiB",flush=True); check("after-weights")
sizes=[578,64,20,6,1,0.25]; hits=0; oom=0
for r in range(R):
    a=[]; k=0; got=0.0
    while got < T:
        s=sizes[k%6]; k+=1
        try: b=torch.empty(int(s*MiB)//2,dtype=torch.bfloat16,device=dev); b.normal_(); a.append(b); got+=s
        except torch.OutOfMemoryError: oom+=1; break
    torch.cuda.synchronize(); hits+=check(f"round {r} full")
    print(f"round {r}: {len(a)} buffers {got:.0f} MiB, free {free()} MiB, reserved {torch.cuda.memory_reserved()//MiB} MiB",flush=True)
    del a
    if r%2: torch.cuda.empty_cache()
    torch.cuda.synchronize(); hits+=check(f"round {r} freed")
print(f"RESULT fixed {T} MiB per round, {R} rounds: {hits} block hits, OOM {oom}",flush=True)
