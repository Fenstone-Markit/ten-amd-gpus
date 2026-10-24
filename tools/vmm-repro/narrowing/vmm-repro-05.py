# vmm-repro-05: does memory ROCm hands to plain hipMalloc after PyTorch releases expandable-segment pages land on a live tensor?
# vmm-repro-03's rounds (fixed MiB per round through PyTorch, empty_cache after odd rounds), plus after each empty_cache:
# hipMalloc up to HM MiB in the same buffer sizes, hipMemset to 0xAB, synchronize, check the blocks, hipFree. A hit reading abababab
# means a plain hipMalloc got a live tensor's memory. Args: TARGET_MiB ROUNDS HM_MiB
import sys, ctypes, torch; MiB=2**20; dev="cuda"; T=int(sys.argv[1]); R=int(sys.argv[2]); HM=int(sys.argv[3])
torch.empty(1,device=dev)
lib=[l.split()[-1] for l in open("/proc/self/maps") if "libamdhip64.so" in l][0]; hip=ctypes.CDLL(lib)
print(torch.__version__, torch.version.hip, lib, flush=True); assert torch.cuda.device_count()==1
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
sizes=[578,64,20,6,1,0.25]; hits=0; hmhits=0; oom=0
for r in range(R):
    a=[]; k=0; got=0.0
    while got<T:
        s=sizes[k%6]; k+=1
        try: b=torch.empty(int(s*MiB)//2,dtype=torch.bfloat16,device=dev); b.normal_(); a.append(b); got+=s
        except torch.OutOfMemoryError: oom+=1; break
    torch.cuda.synchronize(); hits+=check(f"round {r} torch-full")
    del a
    if r%2:
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        ptrs=[]; k=0; hg=0.0
        while hg<HM:
            s=sizes[k%6]; k+=1; p=ctypes.c_void_p(); n=int(s*MiB)
            if hip.hipMalloc(ctypes.byref(p),ctypes.c_size_t(n))!=0: break
            hip.hipMemset(p,0xAB,ctypes.c_size_t(n)); ptrs.append(p); hg+=s
        hip.hipDeviceSynchronize(); h=check(f"round {r} after hipMalloc {len(ptrs)} buffers {hg:.0f} MiB"); hits+=h; hmhits+=h
        print(f"round {r}: hipMalloc {len(ptrs)} buffers {hg:.0f} MiB, free {free()} MiB",flush=True)
        for p in ptrs: hip.hipFree(p)
        hip.hipDeviceSynchronize()
    torch.cuda.synchronize(); hits+=check(f"round {r} freed")
print(f"RESULT {R} rounds, {T} MiB torch + {HM} MiB hipMalloc after each release: {hits} block hits ({hmhits} right after hipMalloc), OOM {oom}",flush=True)
