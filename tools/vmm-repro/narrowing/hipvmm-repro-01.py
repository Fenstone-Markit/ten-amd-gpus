# hipvmm-repro-01: the overwrite without PyTorch, through HIP's virtual-memory API directly (ctypes), shaped like PyTorch's
# expandable segments. A 2 MiB handle holds a known pattern and is mapped at two addresses, A and B. "Weights" and per-round
# "activations" are 20 MiB handles mapped one after another into a separate reserved range; activations are filled with 3f80beef
# and, after odd rounds, unmapped and released (PyTorch's empty_cache). A hit seen at A only means A's mapping changed;
# at A and B both means the memory itself was given to someone else. Args: HEADROOM_MiB ROUNDS HANDLE_TYPE(0 none, 1 posix fd)
import ctypes as C, sys, numpy as np
from ctypes import c_int, c_void_p, c_size_t, c_ubyte, c_ushort, c_ulonglong, byref
H = int(sys.argv[1]); R = int(sys.argv[2]); HT = int(sys.argv[3]); MiB = 2**20
lib = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/libamdhip64.so.7"
hip = C.CDLL(lib)
class Loc(C.Structure): _fields_ = [("type", c_int), ("id", c_int)]
class Prop(C.Structure): _fields_ = [("type", c_int), ("reqHandle", c_int), ("location", Loc), ("win32", c_void_p),
                                     ("compressionType", c_ubyte), ("gpuDirectRDMACapable", c_ubyte), ("usage", c_ushort)]
class Access(C.Structure): _fields_ = [("location", Loc), ("flags", c_int)]
def ok(e, what):
    if e != 0: raise RuntimeError(f"{what} returned {e}")
ok(hip.hipSetDevice(0), "hipSetDevice")
prop = Prop(1, HT, Loc(1, 0), None, 0, 0, 0); acc = Access(Loc(1, 0), 3)
g = c_size_t(); ok(hip.hipMemGetAllocationGranularity(byref(g), byref(prop), 0), "granularity")
def free_mib():
    f = c_size_t(); t = c_size_t(); hip.hipMemGetInfo(byref(f), byref(t)); return f.value // MiB
rt = c_int(); hip.hipRuntimeGetVersion(byref(rt))
print(f"HIP runtime {rt.value}, granularity {g.value} bytes, handle type {HT}, free {free_mib()} MiB", flush=True)
def reserve(size):
    p = c_void_p(); ok(hip.hipMemAddressReserve(byref(p), c_size_t(size), c_size_t(0), None, c_ulonglong(0)), "reserve"); return p.value
def create(size):
    h = c_void_p(); e = hip.hipMemCreate(byref(h), c_size_t(size), byref(prop), c_ulonglong(0)); return h if e == 0 else None
def mapat(addr, size, h):
    ok(hip.hipMemMap(c_void_p(addr), c_size_t(size), c_size_t(0), h, c_ulonglong(0)), "map")
    ok(hip.hipMemSetAccess(c_void_p(addr), c_size_t(size), byref(acc), c_size_t(1)), "setaccess")
# the live block: one 2 MiB handle, mapped at A and at B
SB = 2 * MiB; A = reserve(SB); B = reserve(SB); hb = create(SB); assert hb, "block create failed"
mapat(A, SB, hb); mapat(B, SB, hb)
want = (np.arange(SB // 4, dtype=np.int64) * 7 + 0x5A5A0000).astype(np.uint32)
ok(hip.hipMemcpy(c_void_p(A), want.ctypes.data_as(c_void_p), c_size_t(SB), 1), "H2D"); hip.hipDeviceSynchronize()
def read(addr):
    out = np.empty(SB // 4, dtype=np.uint32); ok(hip.hipMemcpy(out.ctypes.data_as(c_void_p), c_void_p(addr), c_size_t(SB), 2), "D2H"); return out
cur = {"A": want.copy(), "B": want.copy()}
def check(tag):
    n = 0; res = {}
    for name, addr in (("A", A), ("B", B)):
        got = read(addr); d = got != cur[name]
        if d.any():
            n += 1; f = int(d.argmax()); res[name] = f"{int(d.sum())} words, first {got[f:f+2].tobytes().hex()}"; cur[name] = got
    if res: print(f"HIT {tag}: " + "; ".join(f"{k} {v}" for k, v in res.items()), flush=True)
    return n, res
# the big segment: 20 MiB pages mapped one after another, like PyTorch's large-pool expandable segment
PG = 20 * MiB; BIG = reserve(24 * 1024 * MiB); mapped = []
def grow(fill):
    h = create(PG)
    if h is None: return False
    a = BIG + len(mapped) * PG; mapat(a, PG, h); mapped.append((a, h))
    ok(hip.hipMemsetD32(c_void_p(a), c_int(fill), c_size_t(PG // 4)), "memset"); return True
while free_mib() > H + 2048 and grow(0x11111111): pass
nw = len(mapped); hip.hipDeviceSynchronize(); print(f"weights {nw} pages ({nw * 20} MiB), free {free_mib()} MiB", flush=True); check("after weights")
hitsA = hitsB = 0; fails = 0
for r in range(R):
    while free_mib() > H:
        if not grow(0x3F80BEEF): fails += 1; break
    for a, h in mapped[nw:]: hip.hipMemsetD32(c_void_p(a), c_int(0x3F80BEEF), c_size_t(PG // 4))
    hip.hipDeviceSynchronize(); n, res = check(f"round {r} full ({len(mapped) - nw} activation pages, free {free_mib()} MiB)")
    hitsA += "A" in res; hitsB += "B" in res
    if r % 2:
        for a, h in reversed(mapped[nw:]): ok(hip.hipMemUnmap(c_void_p(a), c_size_t(PG)), "unmap"); ok(hip.hipMemRelease(h), "release")
        del mapped[nw:]; hip.hipDeviceSynchronize()
        n, res = check(f"round {r} released"); hitsA += "A" in res; hitsB += "B" in res
print(f"RESULT {R} rounds, headroom {H} MiB, handle type {HT}: hits seen at A {hitsA}, at B {hitsB}; failed creates {fails}", flush=True)
