# glm5_fp8_rdna3: FP8 (e4m3fn, 128x128 block scales) MoE on RDNA3 (gfx1100) through libfp8rdna3.so (fp8-rdna3-lib-01.hip).
# moe(x, w13, s13, w2, s2, topk_weights, topk_ids) -> [T][H], the fused-MoE contract vLLM's experts follow (routed scaling is the
# caller's). Weights must be repacked once with repack_() (bytes per 4 as [b0, b2, b1, b3]). Few tokens: the decode kernels;
# more: the WMMA GEMMs in vLLM's sorted-token format (block 64), SiLU x up and the top-k sum in torch.
import ctypes, os
import torch

_lib = None
BLOCK = 64


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(os.environ.get("GLM5_FP8_LIB", "/fp8/libfp8rdna3.so"))
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        L.fp8r3_moe_decode.argtypes = [P, I, I, P, P, P, P, P, P, P, P, P, I, I, I, I]
        L.fp8r3_moe_gemm.argtypes = [P, I, I, P, P, P, P, P, P, P, P, I, I, I, I, I]
        L.fp8r3_repack.argtypes = [P, P, F]
        for f in (L.fp8r3_moe_decode, L.fp8r3_moe_gemm, L.fp8r3_repack):
            f.restype = I
        _lib = L
    return _lib


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"glm5_fp8_rdna3: {what} failed, HIP error {rc}")


def repack_(w):
    """In place, on a contiguous uint8 (or float8_e4m3fn) CUDA tensor whose size is a multiple of 4."""
    assert w.is_cuda and w.is_contiguous() and w.numel() % 4 == 0
    _check(lib().fp8r3_repack(_stream(), ctypes.c_void_p(w.data_ptr()), w.numel()), "repack")
    return w


def align_torch(topk_ids, num_experts, block=BLOCK):
    """vLLM's moe_align_block_size format, in plain torch (host sync): sorted pair ids padded per expert to the block size
    with the pad value T*topk, one expert id per block, and the padded total."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    M = flat.numel()
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block - 1) // block * block
    starts = torch.cumsum(padded, 0) - padded
    order = torch.argsort(flat, stable=True)
    e_sorted = flat[order]
    first = torch.cumsum(counts, 0) - counts
    rank = torch.arange(M, device=flat.device) - first[e_sorted]
    total = int(padded.sum())
    sorted_ids = torch.full((total,), M, dtype=torch.int32, device=flat.device)
    sorted_ids[starts[e_sorted] + rank] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=flat.device, dtype=torch.int32), padded // block)
    return sorted_ids, expert_ids, torch.tensor([total], dtype=torch.int32, device=flat.device)


def _silu_mul(c1, I):
    try:
        out = torch.empty(c1.shape[0], I, dtype=c1.dtype, device=c1.device)
        torch.ops._C.silu_and_mul(out, c1)
        return out
    except (AttributeError, RuntimeError):
        g, u = c1[:, :I].float(), c1[:, I:].float()
        return (torch.nn.functional.silu(g) * u).to(c1.dtype)


def moe(x, w13, s13, w2, s2, topk_weights, topk_ids, decode_max=8, U=2, align=None):
    """x [T][H] fp16/bf16; w13 [E][2I][H] and w2 [E][H][I] repacked e4m3 (uint8 view); s13 [E][2I/128][H/128], s2 [E][H/128][I/128]
    fp32; topk_weights [T][k]; topk_ids [T][k]. align: a function like vLLM's moe_align_block_size(ids, block, E) -> (sorted, experts, npost);
    defaults to align_torch."""
    T, H = x.shape
    E, I2, _ = w13.shape
    I, topk = I2 // 2, topk_ids.shape[1]
    bf = 1 if x.dtype == torch.bfloat16 else 0
    x = x.contiguous()
    ids = topk_ids.to(torch.int32).contiguous()
    tw = topk_weights.to(torch.float32).contiguous()
    w13b, w2b = w13.view(torch.uint8), w2.view(torch.uint8)
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    L, st = lib(), _stream()
    if T <= decode_max:
        hbuf = torch.empty(T * topk, I, dtype=x.dtype, device=x.device)
        y = torch.empty(T, H, dtype=x.dtype, device=x.device)
        _check(L.fp8r3_moe_decode(st, bf, U, P(w13b), P(s13), P(w2b), P(s2), P(x), P(ids), P(tw), P(hbuf), P(y), T, topk, I, H), "moe_decode")
        return y
    sorted_ids, expert_ids, npost = align_torch(ids, E, BLOCK) if align is None else align(ids, BLOCK, E)
    sorted_ids, expert_ids, npost = sorted_ids.to(torch.int32), expert_ids.to(torch.int32), npost.to(torch.int32)
    M = T * topk
    nb = expert_ids.numel()
    c1 = torch.empty(M, I2, dtype=x.dtype, device=x.device)
    _check(L.fp8r3_moe_gemm(st, bf, 1, P(x), P(w13b), P(s13), P(sorted_ids), P(expert_ids), P(npost), P(tw), P(c1), M, I2, H, topk, nb), "moe_gemm 1")
    h = _silu_mul(c1, I)
    c2 = torch.empty(M, H, dtype=x.dtype, device=x.device)
    _check(L.fp8r3_moe_gemm(st, bf, 2, P(h), P(w2b), P(s2), P(sorted_ids), P(expert_ids), P(npost), P(tw), P(c2), M, H, I, topk, nb), "moe_gemm 2")
    return c2.view(T, topk, H).float().sum(1).to(x.dtype)
