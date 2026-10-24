# ecount-table-01: turn the expert-count snapshots into a per-category, per-layer table. Count only: nothing is pruned
# or changed. CPU only (run inside the GLM container with every GPU hidden, or anywhere torch is installed).
# Args: COUNTS_DIR (holds <tag>-rank<R>.pt and calib-log.jsonl) OUT_DIR
# Each snapshot holds running totals since GLM started, so a category's counts are its snapshot minus the one before.
import glob, json, math, os, sys
import torch
CD, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
E, K = 288, 8
ranks = sorted({int(f.rsplit("-rank", 1)[1][:-3]) for f in glob.glob(os.path.join(CD, "*-rank*.pt"))})
tags = sorted({os.path.basename(f).rsplit("-rank", 1)[0] for f in glob.glob(os.path.join(CD, "*-rank*.pt"))})
tags = [t for t in tags if all(os.path.exists(os.path.join(CD, f"{t}-rank{r}.pt")) for r in ranks)]
snaps = {}
for t in tags:
    merged = {}
    for r in ranks:
        for layer, v in torch.load(os.path.join(CD, f"{t}-rank{r}.pt"), map_location="cpu").items():
            merged[int(layer)] = {"n": v["n"].double(), "tokens": float(v["tokens"])}
    snaps[t] = merged
layers = sorted(snaps[tags[0]])
cats = tags[1:]
# what the driver sent, and what vLLM says it served, per category (from calib-log.jsonl)
sent, metr = {}, {}
for line in open(os.path.join(CD, "calib-log.jsonl")):
    j = json.loads(line)
    if "snapshot" in j: metr[j["snapshot"]] = j["metrics"]
    elif "cat" in j:
        s = sent.setdefault(j["cat"], {"requests": 0, "errors": 0, "prompt": 0, "completion": 0})
        if "error" in j: s["errors"] += 1
        else: s["requests"] += 1; s["prompt"] += j["prompt_tokens"] or 0; s["completion"] += j["completion_tokens"] or 0
def delta(t, prev):
    return {L: {"n": snaps[t][L]["n"] - snaps[prev][L]["n"], "tokens": snaps[t][L]["tokens"] - snaps[prev][L]["tokens"]} for L in layers}
D = {c: delta(c, tags[i]) for i, c in enumerate(cats)}
def stats(n):
    tot = float(n.sum())
    if tot <= 0: return None
    p = n / tot; nz = p[p > 0]
    srt = torch.sort(p, descending=True).values; cum = torch.cumsum(srt, 0)
    return {"picks": tot, "eff": math.exp(float(-(nz * nz.log()).sum())), "k50": int((cum < 0.5).sum()) + 1, "k90": int((cum < 0.9).sum()) + 1,
            "zero": int((n == 0).sum()), "top_share": float(srt[0]), "top_x_uniform": float(srt[0]) * E}
rows, md = [], []
md.append("# GLM-5.3-Flash expert counts by category (count only, nothing pruned)\n")
md.append(f"Snapshots: {', '.join(tags)}. Stages (TP rank 0 of each): {ranks}. MoE layers: {layers[0]} to {layers[-1]} ({len(layers)}). 288 routed experts per layer, 8 picked per token.\n")
md.append("## Tokens: what the counter saw against what was sent\n")
md.append("Counter tokens are tokens the model actually computed; they fall short of vLLM's prompt totals when a prompt prefix was cached and is not recomputed, and run over when padding is counted. Outside traffic means vLLM served more than this driver sent in that window, so that category is mixed with someone else's work.\n")
md.append("| Category | Requests ok / errors | Driver prompt + completion tokens | vLLM prompt + generation tokens | Counter tokens | Picks per token | Outside traffic |")
md.append("| --- | --- | --- | --- | --- | --- | --- |")
prevtag = tags[0]
for c in cats:
    s = sent.get(c.split("-", 1)[1], {"requests": 0, "errors": 0, "prompt": 0, "completion": 0})
    m0, m1 = metr.get(prevtag), metr.get(c)
    if m0 and m1:
        vp = m1["prompt_tokens_total"] - m0["prompt_tokens_total"]; vg = m1["generation_tokens_total"] - m0["generation_tokens_total"]
        vr = m1["request_success_total"] - m0["request_success_total"]
        # the driver sends its requests plus at least one 4-token "Say ok." ping per snapshot; allow 130 pings
        outside = "yes, check" if (vr - s["requests"] - s["errors"]) > 130 or (vp - s["prompt"]) > 130 * 20 else "no"
        vtxt = f"{vp:,.0f} + {vg:,.0f}"
    else:
        outside, vtxt = "unknown", "not logged"
    ct = D[c][layers[0]]["tokens"]; pk = float(D[c][layers[0]]["n"].sum())
    md.append(f"| {c} | {s['requests']} / {s['errors']} | {s['prompt']:,} + {s['completion']:,} | {vtxt} | {ct:,.0f} | {pk / max(ct, 1):.3f} | {outside} |")
    prevtag = c
md.append("\n## Spread of experts per category (averaged over the 42 layers)\n")
md.append("Effective experts: exp(entropy) of the pick distribution, 288 if every expert were used equally. k50 / k90: fewest experts that cover 50 % / 90 % of picks. Unused: experts never picked in that category (in a layer, averaged). Top expert: the busiest expert's share, as a multiple of an even share (1/288). Picks per expert if even: how often each expert would be picked if routing were even; unused counts mean little when this is small.\n")
md.append("| Category | Counter tokens | Picks per expert if even | Effective experts | k50 | k90 | Unused experts | Top expert x even share |")
md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
allN = {L: sum(D[c][L]["n"] for c in cats) for L in layers}
for c in cats + ["all categories"]:
    st = [stats(allN[L] if c == "all categories" else D[c][L]["n"]) for L in layers]; st = [x for x in st if x]
    if not st: md.append(f"| {c} | 0 | 0 | none | none | none | none | none |"); continue
    a = lambda k: sum(x[k] for x in st) / len(st)
    tk = sum(D[x][layers[0]]["tokens"] for x in cats) if c == "all categories" else D[c][layers[0]]["tokens"]
    md.append(f"| {c} | {tk:,.0f} | {tk * K / E:,.0f} | {a('eff'):.1f} | {a('k50'):.1f} | {a('k90'):.1f} | {a('zero'):.1f} | {a('top_x_uniform'):.1f} |")
for c in cats:
    for L in layers:
        x = stats(D[c][L]["n"])
        if x: rows.append({"category": c, "layer": L, **x})
md.append("\n## Per layer, all categories together\n")
md.append("| Layer | Effective experts | k50 | k90 | Unused in every category | Busiest expert (id, x even share) |")
md.append("| --- | --- | --- | --- | --- | --- |")
for L in layers:
    x = stats(allN[L]); i = int(torch.argmax(allN[L]))
    md.append(f"| {L} | {x['eff']:.1f} | {x['k50']} | {x['k90']} | {x['zero']} | {i}, {x['top_x_uniform']:.1f} |")
md.append("\n## How alike the categories are (cosine similarity of pick distributions, averaged over layers; 1.00 = identical)\n")
md.append("| Category | " + " | ".join(c.split("-", 1)[1] for c in cats) + " |")
md.append("| --- | " + " | ".join("---" for _ in cats) + " |")
for a_ in cats:
    cells = []
    for b_ in cats:
        v = [float(torch.nn.functional.cosine_similarity(D[a_][L]["n"], D[b_][L]["n"], dim=0)) for L in layers if D[a_][L]["n"].sum() > 0 and D[b_][L]["n"].sum() > 0]
        cells.append(f"{sum(v) / len(v):.2f}" if v else "n/a")
    md.append(f"| {a_.split('-', 1)[1]} | " + " | ".join(cells) + " |")
md.append("\n## Experts that lean towards one category\n")
md.append("An expert leans towards a category when its share of that category's picks is at least 4x its average share in the other categories, and it was picked at least 200 times there. Counted per layer, summed over layers. This is a description of routing, not a claim that the expert holds that subject.\n")
md.append("| Category | Leaning experts (all layers) | Their share of that category's picks |")
md.append("| --- | --- | --- |")
lean = {}
for c in cats:
    cnt, share = 0, []
    for L in layers:
        n = D[c][L]["n"]; tot = float(n.sum())
        if tot <= 0: continue
        p = n / tot
        others = [D[o][L]["n"] / float(D[o][L]["n"].sum()) for o in cats if o != c and float(D[o][L]["n"].sum()) > 0]
        if not others: continue
        po = sum(others) / len(others)
        m = (p >= 4 * po.clamp(min=1e-9)) & (n >= 200)
        cnt += int(m.sum()); share.append(float(p[m].sum()))
        lean.setdefault(c, {})[L] = [int(i) for i in torch.nonzero(m).flatten()]
    md.append(f"| {c} | {cnt} | {100 * sum(share) / max(len(share), 1):.1f} % per layer on average |")
md.append("\n## Not measured here\n")
md.append("- Routing weights: the weight totals went NaN during GLM's start-up on 4 of 5 stages in this run (fix: claude-ecount-03 drops non-finite values), so this table uses pick counts only.")
md.append("- Whether an unused or rarely used expert matters: counts say how often an expert is picked, not what removing it would cost. That needs a quality test with the expert masked, which is pruning work and is not part of this run.")
md.append("- The shared expert (always on) is not routed, so it is not counted.")
open(os.path.join(OUT, "ecount-table-01.md"), "w").write("\n".join(md) + "\n")
json.dump(rows, open(os.path.join(OUT, "ecount-layers-01.json"), "w"))
json.dump(lean, open(os.path.join(OUT, "ecount-lean-01.json"), "w"))
torch.save({c: {L: D[c][L]["n"] for L in layers} for c in cats}, os.path.join(OUT, "ecount-deltas-01.pt"))
print("\n".join(md))
