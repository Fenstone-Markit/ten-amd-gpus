# mixed-ckpt-01 (rev b: index metadata keeps only total_size): build a mixed GLM-5.3-Flash checkpoint directory: wtdcode's INT4 checkpoint (symlinked file by file, config
# unchanged) plus, for the MoE layers given, the official checkpoint's block-FP8 experts (weight + weight_scale_inv for gate, up
# and down of all 288 experts) copied into one new safetensors file per layer, and an index listing both. vLLM serves it with
# GLM5_FP8_LAYERS=<the same list>: those layers get the RDNA3 FP8 MoE method and their INT4 expert tensors are skipped at load.
# Args: OFFICIAL_DIR WTDCODE_DIR OUT_DIR LAYERS   (LAYERS like "3,6,9" or "3-8"; OUT_DIR must not exist)
import json, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

OFF, WTD, OUT, LAY = sys.argv[1:5]
layers = set()
for part in LAY.split(","):
    if "-" in part:
        a, b = part.split("-"); layers.update(range(int(a), int(b) + 1))
    elif part:
        layers.add(int(part))
layers = sorted(layers)
assert not os.path.exists(OUT), f"{OUT} exists"
os.makedirs(OUT)
wi = json.load(open(os.path.join(WTD, "model.safetensors.index.json")))
oi = json.load(open(os.path.join(OFF, "model.safetensors.index.json")))["weight_map"]
n = 0
for f in sorted(os.listdir(WTD)):
    if f == "model.safetensors.index.json":
        continue
    os.symlink(os.path.relpath(os.path.join(WTD, f), OUT), os.path.join(OUT, f)); n += 1
print(f"{n} files of the INT4 checkpoint linked", flush=True)
wm = dict(wi["weight_map"])
total = 0
for L in layers:
    t0 = time.time()
    pre = f"model.language_model.layers.{L}.mlp.experts."
    names = sorted(k for k in oi if k.startswith(pre) and (k.endswith(".weight") or k.endswith(".weight_scale_inv")))
    assert len(names) == 288 * 6, f"layer {L}: {len(names)} expert tensors in the official checkpoint, expected {288 * 6}"
    clash = [k for k in names if k in wm]
    assert not clash, f"layer {L}: names already in the INT4 index, e.g. {clash[:2]}"
    byfile = {}
    for k in names:
        byfile.setdefault(oi[k], []).append(k)
    out = {}
    for f, ks in byfile.items():
        with safe_open(os.path.join(OFF, f), "pt") as h:
            for k in ks:
                out[k] = h.get_tensor(k)
    bad = [k for k, v in out.items() if (k.endswith(".weight") and v.dtype != torch.float8_e4m3fn) or (k.endswith("_inv") and v.dtype != torch.float32)]
    assert not bad, f"layer {L}: unexpected dtypes, e.g. {bad[:2]} {out[bad[0]].dtype}"
    fn = f"fp8-experts-layer{L:02d}.safetensors"
    save_file(out, os.path.join(OUT, fn), metadata={"format": "pt"})
    sz = os.path.getsize(os.path.join(OUT, fn)); total += sz
    for k in names:
        wm[k] = fn
    print(f"layer {L}: {len(names)} tensors from {len(byfile)} file(s), {sz / 2**30:.2f} GiB, {time.time() - t0:.0f} s", flush=True)
    del out
meta = dict(wi.get("metadata", {}))
if "total_size" in meta:
    meta["total_size"] = int(meta["total_size"]) + total  # file bytes, close enough (the INT4 tensors of these layers stay listed)
idx = {"metadata": meta, "weight_map": wm}
json.dump(idx, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
open(os.path.join(OUT, "MIXED.md"), "w").write(f"# GLM-5.3-Flash mixed checkpoint\n\nINT4 base: {WTD}\nFP8 experts from: {OFF}\nFP8 expert layers: {LAY} ({len(layers)} layers)\nServe with GLM5_FP8_LAYERS={LAY}\n")
print(f"DONE: {len(layers)} FP8 expert layers, {total / 2**30:.1f} GiB new, index {len(wm)} tensors", flush=True)
