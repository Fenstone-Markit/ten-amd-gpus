# mqa-test-02.py: which is closer to exact fp32 maths, the torch fallback or the Triton kernel? (one card, GLM stopped)
import sys, torch
sys.path.insert(0, "/work")
from mqa_logits_rdna3 import fp8_mqa_logits_rdna3
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch
torch.manual_seed(1); dev = "cuda"; f8 = torch.float8_e4m3fn
M, N, H, D = 1152, 10000, 32, 128
q = (torch.randn(M, H, D, device=dev) * 0.5).to(f8); k = (torch.randn(N, D, device=dev) * 0.5).to(f8)
sc = torch.rand(N, 1, device=dev) * 0.05 + 0.01; w = torch.randn(M, H, device=dev) * 0.1
ks = torch.zeros(M, dtype=torch.int32, device=dev); ke = torch.full((M,), N, dtype=torch.int32, device=dev)
exact = torch.zeros(M, N, device=dev, dtype=torch.float64)
for h in range(H):
    s = (q[:, h].double() @ k.double().T) * sc.reshape(-1).double()
    exact += s.clamp(min=0) * w[:, h].double()[:, None]
ref = fp8_mqa_logits_torch(q, (k, sc), w, ks, ke).double(); new = fp8_mqa_logits_rdna3(q, (k, sc), w, ks, ke).double()
for name, x in (("torch fallback", ref), ("triton kernel", new)):
    e = (x - exact).abs()
    ov = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(torch.topk(x[-64:], 512, 1).indices, torch.topk(exact[-64:], 512, 1).indices)) / (64 * 512)
    print(f"{name}: max abs error vs exact {e.max().item():.2e}, mean {e.mean().item():.2e}, top-512 overlap with exact {ov * 100:.2f}%")
