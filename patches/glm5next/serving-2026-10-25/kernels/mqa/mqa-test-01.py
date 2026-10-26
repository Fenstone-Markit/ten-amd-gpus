# mqa-test-01.py: compare fp8_mqa_logits_rdna3 with the in-tree torch fallback (values and top-k sets) and time both.
# Runs on ONE card while GLM is stopped. Shapes as the indexer's prompt chunks: M query rows, H=32 heads, D=128, N pooled keys.
import sys, time, torch
sys.path.insert(0, "/work")
from mqa_logits_rdna3 import fp8_mqa_logits_rdna3
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch
torch.manual_seed(0); dev = "cuda"; f8 = torch.float8_e4m3fn
def case(M, N, H=32, D=128, causal=True, topk=512):
    q = (torch.randn(M, H, D, device=dev) * 0.5).to(f8)
    k = (torch.randn(N, D, device=dev) * 0.5).to(f8)
    sc = torch.rand(N, 1, device=dev) * 0.05 + 0.01
    w = torch.randn(M, H, device=dev) * 0.1
    if causal:
        ke = (torch.arange(M, device=dev) * N // M + N // M).clamp(max=N).to(torch.int32); ks = torch.zeros(M, dtype=torch.int32, device=dev)
    else:
        ks = torch.zeros(M, dtype=torch.int32, device=dev); ke = torch.full((M,), N, dtype=torch.int32, device=dev)
    ref = fp8_mqa_logits_torch(q, (k, sc), w, ks, ke)
    out = fp8_mqa_logits_rdna3(q, (k, sc), w, ks, ke)
    fin = torch.isfinite(ref)
    same_inf = bool(((~torch.isfinite(out)) == (~fin)).all())
    err = (out[fin] - ref[fin]).abs().max().item(); mag = ref[fin].abs().max().item()
    kk = min(topk, int(ke.min().item())) if causal else topk
    rows = torch.arange(M - 64, M, device=dev)  # long rows
    a = torch.topk(ref[rows], kk, dim=1).indices.sort(dim=1).values; b = torch.topk(out[rows], kk, dim=1).indices.sort(dim=1).values
    overlap = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)) / (64 * kk)
    def tm(f):
        f(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / 5 * 1000
    t_ref = tm(lambda: fp8_mqa_logits_torch(q, (k, sc), w, ks, ke)); t_new = tm(lambda: fp8_mqa_logits_rdna3(q, (k, sc), w, ks, ke))
    print(f"M={M} N={N} causal={causal}: -inf pattern same={same_inf}, max abs err {err:.2e} (max |logit| {mag:.2e}), "
          f"top-{kk} overlap {overlap * 100:.2f}%, torch {t_ref:.1f} ms, triton {t_new:.1f} ms ({t_ref / t_new:.1f}x)", flush=True)
for M, N, c in [(256, 1000, False), (1152, 2500, True), (1152, 10000, True), (1152, 29100, True), (1000, 29100, False)]:
    case(M, N, causal=c)
