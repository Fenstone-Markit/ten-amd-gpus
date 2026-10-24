# hipvmm-replay-01: replay a hipmem-trace-01 call trace (PyTorch's own reserve/create/map/set-access/unmap/release sequence) in pure HIP.
# The first mapped page holds a known pattern and its handle is also mapped at a second address B. Every set-access range after that is
# filled with 3f80beef (PyTorch would write data there). The block is checked at A and B after every burst of same-kind calls.
# At the first create that failed in the trace, a plain hipMalloc ballast takes all but BALLAST_LEFT MiB so creates fail here too;
# a create whose outcome differs from the trace is counted. Args: TRACE BALLAST_LEFT_MiB
import ctypes as C, sys, numpy as np
from ctypes import c_int, c_void_p, c_size_t, c_ubyte, c_ushort, c_ulonglong, byref
TR = sys.argv[1]; LEFT = int(sys.argv[2]); MiB = 2**20
hip = C.CDLL("/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/libamdhip64.so.7")
class Loc(C.Structure): _fields_ = [("type", c_int), ("id", c_int)]
class Prop(C.Structure): _fields_ = [("type", c_int), ("reqHandle", c_int), ("location", Loc), ("win32", c_void_p),
                                     ("compressionType", c_ubyte), ("gpuDirectRDMACapable", c_ubyte), ("usage", c_ushort)]
class Access(C.Structure): _fields_ = [("location", Loc), ("flags", c_int)]
def ok(e, what):
    if e != 0: raise RuntimeError(f"{what} returned {e}")
ok(hip.hipSetDevice(0), "hipSetDevice"); acc = Access(Loc(1, 0), 3)
def free_mib():
    f = c_size_t(); t = c_size_t(); hip.hipMemGetInfo(byref(f), byref(t)); return f.value // MiB
ops = [l.split() for l in open(TR) if l.strip()]
base = {}; hmap = {}; A = None; B = None; SB = 2 * MiB
want = (np.arange(SB // 4, dtype=np.int64) * 7 + 0x5A5A0000).astype(np.uint32); cur = {}
def tr(addr):
    a = int(addr, 16)
    for ob, (nb, sz) in base.items():
        if ob <= a < ob + sz: return nb + (a - ob)
    raise RuntimeError(f"address {addr} outside every reserved range")
def read(addr):
    out = np.empty(SB // 4, dtype=np.uint32); ok(hip.hipMemcpy(out.ctypes.data_as(c_void_p), c_void_p(addr), c_size_t(SB), 2), "D2H"); return out
hitsA = hitsB = 0; checks = 0; mism = 0; ballast = None; skipped = set()
def check(where):
    global hitsA, hitsB, checks
    if A is None or "A" not in cur: return
    hip.hipDeviceSynchronize(); checks += 1; res = {}
    for name, addr in (("A", A), ("B", B)):
        got = read(addr); d = got != cur[name]
        if d.any():
            f = int(d.argmax()); res[name] = f"{int(d.sum())} words, first {got[f:f+2].tobytes().hex()}"; cur[name] = got
    if res:
        hitsA += "A" in res; hitsB += "B" in res; print(f"HIT after trace line {where}: " + "; ".join(f"{k} {v}" for k, v in res.items()), flush=True)
prev = None
for i, o in enumerate(ops, 1):
    k = o[0]; failed = any(x.startswith("e=") for x in o)
    if prev is not None and k != prev: check(i - 1)
    prev = k
    if k == "V":
        if failed: continue
        ob, sz = int(o[1], 16), int(o[2])
        if ob in base: continue
        p = c_void_p(); ok(hip.hipMemAddressReserve(byref(p), c_size_t(sz), c_size_t(0), c_void_p(ob), c_ulonglong(0)), "reserve")
        base[ob] = (p.value, sz); print(f"reserve {sz >> 20} MiB at {hex(p.value)} (trace {o[1]})", flush=True)
    elif k == "C":
        sz = int(o[2]); kk = [int(x) for x in o[3:7]]
        if failed and ballast is None:
            ballast = []
            for chunk in (64, 2):
                while free_mib() - chunk >= LEFT:
                    q = c_void_p()
                    if hip.hipMalloc(byref(q), c_size_t(chunk * MiB)) != 0: break
                    ballast.append((q, chunk))
            print(f"ballast {sum(c for _, c in ballast)} MiB in {len(ballast)} pieces at trace line {i}, free now {free_mib()} MiB", flush=True)
        prop = Prop(kk[0], kk[1], Loc(kk[2], kk[3]), None, 0, 0, 0); h = c_void_p()
        e = hip.hipMemCreate(byref(h), c_size_t(sz), byref(prop), c_ulonglong(0))
        if (e == 0) == failed: mism += 1
        if e == 0 and failed: hip.hipMemRelease(h)
        elif e == 0: hmap[o[1]] = h
        elif not failed: skipped.add(o[1])
    elif k == "M":
        if o[4] in skipped: continue
        a = tr(o[1]); ok(hip.hipMemMap(c_void_p(a), c_size_t(int(o[2])), c_size_t(int(o[3])), hmap[o[4]], c_ulonglong(0)), f"map line {i}")
        if A is None:
            A = a; pb = c_void_p(); ok(hip.hipMemAddressReserve(byref(pb), c_size_t(SB), c_size_t(0), None, c_ulonglong(0)), "reserve B"); B = pb.value
            ok(hip.hipMemMap(c_void_p(B), c_size_t(SB), c_size_t(0), hmap[o[4]], c_ulonglong(0)), "map B")
            ok(hip.hipMemSetAccess(c_void_p(B), c_size_t(SB), byref(acc), c_size_t(1)), "access B")
    elif k == "S":
        a = tr(o[1]); s = int(o[2])
        if hip.hipMemSetAccess(c_void_p(a), c_size_t(s), byref(acc), c_size_t(1)) != 0: mism += 1; continue
        if a == A and A not in cur:
            ok(hip.hipMemcpy(c_void_p(A), want.ctypes.data_as(c_void_p), c_size_t(SB), 1), "H2D"); hip.hipDeviceSynchronize()
            cur["A"] = want.copy(); cur["B"] = read(B); cur[A] = 1
        else:
            ok(hip.hipMemsetD32(c_void_p(a), c_int(0x3F80BEEF), c_size_t(s // 4)), "fill")
    elif k == "U":
        if failed: continue
        ok(hip.hipMemUnmap(c_void_p(tr(o[1])), c_size_t(int(o[2]))), f"unmap line {i}")
    elif k == "R":
        if o[1] in skipped or o[1] not in hmap: continue
        ok(hip.hipMemRelease(hmap.pop(o[1])), f"release line {i}")
check(len(ops))
print(f"RESULT replayed {len(ops)} calls, {checks} checks: hits seen at A {hitsA}, at B {hitsB}; creates that differed from the trace {mism}; handles skipped {len(skipped)}", flush=True)
