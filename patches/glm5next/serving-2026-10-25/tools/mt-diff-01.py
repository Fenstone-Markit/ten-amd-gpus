#!/usr/bin/env python3
# mt-diff-01.py <dir>: subtract the module-timing tables mt-pp<N>-before.txt from mt-pp<N>-after.txt (cumulative counters)
# and print, per stage, where the stage's prompt-step time went, grouped into the parts of a layer.
import re, sys
D = sys.argv[1]
def load(f):
    L = open(f).read().splitlines(); m = re.match(r"(\d+) prompt steps, (\d+) tokens, stage wall time (\d+) ms", L[0])
    rows = {}
    for l in L[3:]:
        c = [x.strip() for x in l.strip("|").split("|")]
        rows[c[0]] = float(c[1])
    return int(m.group(1)), int(m.group(2)), float(m.group(3)), rows
G = [("experts (MoE, routed + shared)", ["Glm5NextMoE:mlp"]), ("dense MLP (first layers)", ["Glm5NextMLP:mlp"]),
     ("mHC residual mixing", ["MHCFusedPostPreOp:mhc_fused_post_pre_op", "MHCPreOp:mhc_pre_op", "MHCPostOp:mhc_post_op"]),
     ("sparse indexer (MLA layers)", ["Indexer:self_attn.indexer"]),
     ("MLA attention core", ["MLAAttention:self_attn.mla_attn.mla_attn"]),
     ("KDA linear attention (incl. its o_proj)", ["Glm5NextLinearAttention:self_attn"])]
print("| stage | tokens | wall ms | " + " | ".join(g for g, _ in G) + " | rest of MLA layers | outside the layers |")
print("| --- " * (len(G) + 5) + "|")
for p in (0, 1, 2):
    s0, t0, w0, r0 = load(f"{D}/mt-pp{p}-before.txt"); s1, t1, w1, r1 = load(f"{D}/mt-pp{p}-after.txt")
    d = {k: r1.get(k, 0) - r0.get(k, 0) for k in r1}; wall = w1 - w0; tok = t1 - t0
    vals = [sum(d.get(k, 0) for k in ks) for _, ks in G]
    mla = d.get("Glm5NextMLAAttention:self_attn", 0) - d.get("Indexer:self_attn.indexer", 0) - d.get("MLAAttention:self_attn.mla_attn.mla_attn", 0)
    outside = wall - d.get("Glm5NextDecoderLayer:(whole layer)", 0)
    cells = [f"{v:.0f} ({100 * v / wall:.0f}%)" for v in vals + [mla, outside]]
    print(f"| {p + 1} | {tok} | {wall:.0f} | " + " | ".join(cells) + " |")
    print(f"<!-- stage {p + 1}: o_proj {d.get('RowParallelLinear:self_attn.o_proj', 0):.0f} ms; layer sum check {d.get('Glm5NextDecoderLayer:(whole layer)', 0) - sum(vals) - mla:.0f} ms unaccounted inside layers -->")
