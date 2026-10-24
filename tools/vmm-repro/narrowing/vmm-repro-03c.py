# vmm-repro-03c: vmm-repro-03 run under hipmem-watch-02, which maps the first small-pool page (blocks 0 to 2) at a second address B.
# Each check reads blocks 0 to 2 through PyTorch's address (A) and through B. Hit at A only: A's mapping changed. At both: the memory changed.
# vmm-repro-03: like vmm-repro.py, but each round allocates a fixed byte total (TARGET MiB, the per-round fill seen on the control stack),
# so pressure does not depend on what mem_get_info reports. empty_cache after odd rounds. Prints every round. Args: TARGET_MiB ROUNDS
import sys, ctypes, numpy as np, torch; MiB=2**20; dev="cuda"; T=int(sys.argv[1]); R=int(sys.argv[2])
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
lib=[l.split()[-1] for l in open("/proc/self/maps") if "libamdhip64.so" in l][0]; hip=ctypes.CDLL(lib)
W=ctypes.CDLL("/w/hipmem-watch-02.so"); W.hipmem_watch_a.restype=W.hipmem_watch_b.restype=ctypes.c_void_p; W.hipmem_watch_size.restype=ctypes.c_size_t
WA, WB, WS = W.hipmem_watch_a(), W.hipmem_watch_b(), W.hipmem_watch_size()
print(f"watched page A {hex(WA or 0)} size {WS}, second mapping B {hex(WB or 0)}; blocks at " + " ".join(hex(c[0].data_ptr()) for c in can), flush=True)
assert WB, "no second mapping"
onpage=[i for i,c in enumerate(can) if WA <= c[0].data_ptr() < WA+WS]; print(f"blocks on the watched page: {onpage}", flush=True)
expB={i: can[i][1].copy() for i in onpage}
def readB(i):
    o=np.empty(131072,dtype=np.int32); ok=hip.hipMemcpy(o.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(WB + can[i][0].data_ptr() - WA), ctypes.c_size_t(524288), 2); assert ok==0; return o
nA=[0]; nB=[0]
def checkB(tag):
    torch.cuda.synchronize()
    for i in onpage:
        g=readB(i); d=g!=expB[i]
        if d.any(): nB[0]+=1; f=int(d.argmax()); print(f"HIT-B {tag} block {i}: {int(d.sum())} words, first bytes {g[f:f+4].tobytes().hex()}",flush=True); expB[i]=g.copy()
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
    torch.cuda.synchronize(); hits+=check(f"round {r} full"); checkB(f"round {r} full")
    print(f"round {r}: {len(a)} buffers {got:.0f} MiB, free {free()} MiB, reserved {torch.cuda.memory_reserved()//MiB} MiB",flush=True)
    del a
    if r%2: torch.cuda.empty_cache()
    torch.cuda.synchronize(); hits+=check(f"round {r} freed"); checkB(f"round {r} freed")
print(f"RESULT fixed {T} MiB per round, {R} rounds: {hits} block hits through A (all four blocks), {nB[0]} through B (blocks {onpage}), OOM {oom}",flush=True)
