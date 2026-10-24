# [claude-ecount-02] Read-only count of which routed experts GLM uses, per MoE layer (GLM5_EXPERT_COUNT=1).
# Counters are non-persistent buffers made when the layer is built (so torch.compile and CUDA graphs see fixed tensors);
# observe() runs inside the compiled forward: GLM's routing rule (sigmoid scores, bias-corrected top-k, weights renormalized,
# n_group 1) gives the chosen experts, and in-place adds count 1 per (token, expert) and the routing weight. The model's own
# routing and output are untouched. gpu_worker saves the counters when a trigger file appears.
# (-01 called vLLM's router lazily and used .item(), which breaks vLLM's single-graph torch.compile.)
import re
import torch

def layer_of(prefix):
    m = re.search(r"layers\.(\d+)", prefix or "")
    return int(m.group(1)) if m else -1

def init(mod, n_experts, layer, k):
    mod._ec_layer = layer; mod._ec_k = k
    mod.register_buffer("_ec_n", torch.zeros(n_experts, dtype=torch.float32), persistent=False)
    mod.register_buffer("_ec_w", torch.zeros(n_experts, dtype=torch.float32), persistent=False)
    mod.register_buffer("_ec_tok", torch.zeros(1, dtype=torch.float32), persistent=False)

def route(logits, bias, k):
    s = torch.sigmoid(logits.float())
    choice = s + bias.float() if bias is not None else s
    ids = torch.topk(choice, k, dim=-1).indices
    w = s.gather(1, ids)
    return w / w.sum(-1, keepdim=True), ids

def observe(mod, logits, bias):
    w, ids = route(logits, bias, mod._ec_k)
    ids = ids.reshape(-1)
    mod._ec_n.index_add_(0, ids, torch.ones_like(w.reshape(-1)))
    mod._ec_w.index_add_(0, ids, w.reshape(-1))
    mod._ec_tok.add_(logits.shape[0])

def snapshot(model):
    out = {}
    for m in model.modules():
        if hasattr(m, "_ec_n"):
            out[int(m._ec_layer)] = {"n": m._ec_n.float().cpu(), "w": m._ec_w.float().cpu(), "tokens": float(m._ec_tok.float().cpu()[0])}
    return out
