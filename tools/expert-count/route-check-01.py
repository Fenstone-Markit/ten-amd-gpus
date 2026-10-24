# route-check-01: compare glm5_ecount.route (GLM's rule in plain torch) with vLLM's GPU topk_sigmoid kernel on random
# router logits with a correction bias, 288 experts, top-8, renormalized. Reports the share of tokens choosing the same 8.
import sys, torch; sys.path.insert(0, "/w"); import glm5_ecount as ec
import vllm._custom_ops as ops
print("vllm", __import__("vllm").__version__, "| ops.topk_sigmoid:", hasattr(ops, "topk_sigmoid"))
torch.manual_seed(0); T, E, K = 20000, 288, 8
logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16) * 2
bias = (torch.randn(E, device="cuda", dtype=torch.float32) * 0.05)
w1, i1 = ec.route(logits, bias, K)
tw = torch.empty(T, K, device="cuda", dtype=torch.float32); ti = torch.empty(T, K, device="cuda", dtype=torch.int32); tei = torch.empty(T, K, device="cuda", dtype=torch.int32)
import inspect; print("topk_sigmoid signature:", inspect.signature(ops.topk_sigmoid))
ops.topk_sigmoid(tw, ti, tei, logits, True, bias)
same = (torch.sort(i1.long(), -1).values == torch.sort(ti.long(), -1).values).all(-1).float().mean().item()
o1 = torch.sort(i1.long(), -1); o2 = torch.sort(ti.long(), -1)
wd = (w1.gather(1, o1.indices) - tw.gather(1, o2.indices)).abs().max().item()
print(f"tokens with the same 8 experts: {100 * same:.3f}% of {T}; max routing-weight difference where equal: {wd:.2e}")
