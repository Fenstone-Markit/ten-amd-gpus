# mtp-ckpt-01: build glm53-mixed-e4 = glm53-mixed-e3 (linked entry by entry, nothing copied) plus the MTP draft layer's (45)
# routed experts in block FP8 from the official checkpoint, in one new file fp8-experts-layer45.safetensors. In e3 those
# experts are BF16 (wtdcode left the draft layer unquantised: 14.5 GB, too big for the last stage); in e4 the index sends
# layer 45's gate/up/down expert weights to the FP8 file and adds their weight_scale_inv. Everything else of layer 45
# (attention, shared expert, norms, eh_proj) stays BF16 from e3. Serve with 45 in GLM5_FP8_LAYERS.
# Args: OFFICIAL_DIR E3_DIR OUT_DIR   (OUT_DIR must not exist). CPU only.
import json, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
OFF, E3, OUT = sys.argv[1:4]
assert not os.path.exists(OUT), f"{OUT} exists"
os.makedirs(OUT)
n = 0
for f in sorted(os.listdir(E3)):
    if f in ("model.safetensors.index.json", "MIXED.md"):
        continue
    src = os.path.join(E3, f)
    tgt = os.readlink(src) if os.path.islink(src) else os.path.join("..", os.path.basename(E3), f)
    os.symlink(tgt, os.path.join(OUT, f)); n += 1
bad = [f for f in os.listdir(OUT) if not os.path.exists(os.path.join(OUT, f))]
assert not bad, f"dangling links: {bad[:3]}"
print(f"{n} entries of e3 linked", flush=True)
ei = json.load(open(os.path.join(E3, "model.safetensors.index.json")))
oi = json.load(open(os.path.join(OFF, "model.safetensors.index.json")))["weight_map"]
wm = dict(ei["weight_map"])
pre = "model.language_model.layers.45.mlp.experts."
names = sorted(k for k in oi if k.startswith(pre) and (k.endswith(".weight") or k.endswith(".weight_scale_inv")))
assert len(names) == 288 * 6, f"{len(names)} layer-45 expert tensors in the official checkpoint, expected {288 * 6}"
old = sorted(k for k in wm if k.startswith(pre))
assert len(old) == 288 * 3 and all(wm[k].startswith("model-nextn-") for k in old), "e3's layer-45 experts are not the 864 BF16 nextn tensors"
assert set(old) <= set(names), "official and e3 layer-45 expert weight names differ"
t0 = time.time(); byfile = {}
for k in names: byfile.setdefault(oi[k], []).append(k)
out = {}
for f, ks in byfile.items():
    with safe_open(os.path.join(OFF, f), "pt") as h:
        for k in ks: out[k] = h.get_tensor(k)
bad = [k for k, v in out.items() if (k.endswith(".weight") and v.dtype != torch.float8_e4m3fn) or (k.endswith("_inv") and v.dtype != torch.float32)]
assert not bad, f"unexpected dtypes, e.g. {bad[:2]}"
shapes = {k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1]: tuple(out[k].shape) for k in names[:6]}
fn = "fp8-experts-layer45.safetensors"
save_file(out, os.path.join(OUT, fn), metadata={"format": "pt"})
sz = os.path.getsize(os.path.join(OUT, fn))
for k in names: wm[k] = fn
meta = dict(ei.get("metadata", {}))
if "total_size" in meta: meta["total_size"] = int(meta["total_size"]) + sz
json.dump({"metadata": meta, "weight_map": wm}, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
open(os.path.join(OUT, "MIXED.md"), "w").write(
    "# GLM-5.3-Flash mixed checkpoint e4\n\n= glm53-mixed-e3 (INT4 base + FP8 experts of layers 3,6,...,42) plus the MTP draft layer 45's\n"
    f"routed experts in block FP8 from {OFF}.\nServe with 45 added to GLM5_FP8_LAYERS (draft layer on our FP8 kernels).\n")
print(f"layer 45: {len(names)} tensors from {len(byfile)} file(s), {sz / 2**30:.2f} GiB, {time.time() - t0:.0f} s; shapes {shapes}", flush=True)
print(f"DONE: index {len(wm)} tensors", flush=True)
