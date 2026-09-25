"""gate2.py — batched 3D attention gate (unlike gate.py, which runs ONE sequence).

Why gate2: gate.py runs num_seqs=1, so the 3D segment buffer (first dim =
seq_threshold_3D = 128 // num_kv_heads) is never indexed past slot 128. A real
speculative-verify batch of S sequences with ntok query tokens produces ntok*S
query tokens, and the 3D kernel indexes the segment buffers by the GLOBAL
query-token index. So ntok*S > 128 (i.e. S>64 at 2 tokens) writes out of bounds
(hipErrorIllegalAddress). This gate builds those batches and checks, per case:
  1. did the 3D path actually RUN (grid has 3 dims) or fall back to 2D?
  2. is the 3D output correct vs the 2D reference?
A case that "passes" only because it fell back to 2D is marked "fell back".

It calls the REAL wrapper `unified_attention` (not the raw kernel), so the live
dispatch gate (triton_unified_attention.py:929) is exercised. The segm buffers are
allocated exactly like the serving path (triton_attn.py:181-200), first dim =
MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads (= 128 for our 1-KV-head shapes).

Two modes:
  python3 gate2.py --all           # run the full matrix, one subprocess per case
  python3 gate2.py --nq 6 --hs 256 --nkv 1 --ctx 2048 --seqs 65 --ntok 2   # one case
"""
import os
UUIDS = {
    "G2": "GPU-cbfe8a265e29739c",
    "G9": "GPU-fec7cca764f40561",
    "G8": "GPU-1b9fb8181ce9991e",
    "G10": "GPU-a25cdb16a49e2dcf",
}
_gpu = os.environ.get("GATE_GPU", "G2")
assert _gpu in UUIDS, f"GATE_GPU must be one of {list(UUIDS)}"
_uuid = UUIDS[_gpu]
os.environ["ROCR_VISIBLE_DEVICES"] = _uuid
os.environ["HIP_VISIBLE_DEVICES"] = _uuid
os.environ["CUDA_VISIBLE_DEVICES"] = _uuid

import argparse
import json
import math
import subprocess
import sys

import triton
import torch


DEVICE = "cuda"
BLOCK_SIZE = 16
NUM_SEGMENTS = 16
TILE_SIZE_DECODE = 16
DTYPE = torch.bfloat16
TOL_ABS = 0.02
TOL_REL = 0.05
REL_FLOOR = 0.01
MIN_LAUNCH_GRID_SIZE_2D = 128


def cdiv(a, b):
    return (a + b - 1) // b


def build_batched(head_size, num_kv_heads, nq, context_len, num_seqs,
                  num_query_tokens, seed=1234):
    """S sequences, each with num_query_tokens query tokens (the LAST positions of
    a length-(context_len+num_query_tokens) seq). Uniform batch. Mirrors the
    serving-path paged-KV layout used by triton_attn (block_table, cu_seqlens_q,
    seq_lens)."""
    num_query_heads = num_kv_heads * nq
    total_kv_per_seq = context_len + num_query_tokens
    blocks_per_seq = cdiv(total_kv_per_seq, BLOCK_SIZE)
    total_blocks = blocks_per_seq * num_seqs
    g = torch.Generator(device="cpu").manual_seed(seed)

    k = torch.randn(total_blocks, BLOCK_SIZE, num_kv_heads, head_size,
                    generator=g, dtype=torch.float32)
    v = torch.randn(total_blocks, BLOCK_SIZE, num_kv_heads, head_size,
                    generator=g, dtype=torch.float32)
    for s in range(num_seqs):
        base = s * blocks_per_seq
        for b in range(base, base + blocks_per_seq):
            off = total_kv_per_seq - (b - base) * BLOCK_SIZE
            k[b, off:] = 0.0
            v[b, off:] = 0.0
    k = k.to(DEVICE, DTYPE)
    v = v.to(DEVICE, DTYPE)

    block_table = torch.arange(total_blocks, device=DEVICE,
                               dtype=torch.int32).view(num_seqs, blocks_per_seq)
    cu = [0]
    for _ in range(num_seqs):
        cu.append(cu[-1] + num_query_tokens)
    cu_seqlens_q = torch.tensor(cu, device=DEVICE, dtype=torch.int32)
    seqused_k = torch.full((num_seqs,), total_kv_per_seq, device=DEVICE, dtype=torch.int32)
    total_q = num_seqs * num_query_tokens
    q = torch.randn(total_q, num_query_heads, head_size,
                    generator=g, dtype=torch.float32).to(DEVICE, DTYPE)
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_size)
    return {
        "q": q, "k": k, "v": v, "out": out,
        "block_table": block_table, "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k, "num_query_heads": num_query_heads,
        "num_queries_per_kv": nq, "num_seqs": num_seqs,
        "head_size": head_size, "head_size_padded": triton.next_power_of_2(head_size),
        "scale": scale, "num_query_tokens": num_query_tokens,
        "context_len": context_len, "total_q": total_q,
    }


def segm_buffers(bufdim, num_query_heads, hsp):
    """Sized exactly like triton_attn.py:181-200 (serving path)."""
    so = torch.zeros(bufdim, num_query_heads, NUM_SEGMENTS, hsp, device=DEVICE, dtype=torch.float32)
    sm = torch.full((bufdim, num_query_heads, NUM_SEGMENTS), float("-inf"), device=DEVICE, dtype=torch.float32)
    se = torch.zeros(bufdim, num_query_heads, NUM_SEGMENTS, device=DEVICE, dtype=torch.float32)
    return so, sm, se


def make_grid_spy():
    """Wrap the module-level kernel so we can read the launch grid of the main
    attention kernel (3 dims => 3D ran; 2 dims => fell back to 2D)."""
    import vllm.v1.attention.ops.triton_unified_attention as T
    orig = T.kernel_unified_attention
    captured = []

    class Spy:
        def __getitem__(self, grid):
            def call(*a, **k):
                captured.append(tuple(grid))
                return orig[grid](*a, **k)
            return call
    T.kernel_unified_attention = Spy()
    return captured, T


def run_wrapper(inp, use_3d, bufdim, seq_threshold_3D):
    """Call the real unified_attention wrapper. use_3d=False forces the 2D reference
    path (pass None for the 3D params); use_3d=True passes the serving-sized segm
    buffers so the wrapper may select 3D.

    seq_threshold_3D drives the DISPATCH gate (num_seqs > seq_threshold_3D) and is
    the real serving value (128 // nkv), UNCHANGED by the buffer fix. bufdim is the
    FIRST DIMENSION of the segm buffers, which the fix changes from seq_threshold_3D
    to 2*seq_threshold_3D. Keeping them separate models the real serving path.
    """
    import vllm.v1.attention.ops.triton_unified_attention as T
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    q, k, v, out = inp["q"], inp["k"], inp["v"], inp["out"]
    nkv = k.shape[2]
    if use_3d:
        so, sm, se = segm_buffers(bufdim, inp["num_query_heads"], inp["head_size_padded"])
        nseg = NUM_SEGMENTS
    else:
        so = sm = se = None
        seq_threshold_3D = None
        nseg = None
    captured, _ = make_grid_spy()
    unified_attention(
        q, k, v, out,
        inp["cu_seqlens_q"], inp["num_query_tokens"], inp["seqused_k"],
        inp["context_len"] + inp["num_query_tokens"],
        inp["scale"], True, (-1, -1),
        inp["block_table"], 0.0, None, None, None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=nseg,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
    )
    torch.cuda.synchronize()
    grid = captured[0] if captured else None
    return out.clone(), grid


def compare(ref, cand):
    d = (cand.float() - ref.float()).abs()
    max_abs = float(d.max().item())
    mask = ref.float().abs() >= REL_FLOOR
    has_rel = bool(mask.any())
    max_rel = float((d[mask] / ref.float()[mask].abs()).max().item()) if has_rel else float("nan")
    rel_ok = (not has_rel) or (max_rel <= TOL_REL)
    return {"max_abs": max_abs, "max_rel": max_rel, "has_rel": has_rel,
            "pass": bool((max_abs <= TOL_ABS) and rel_ok)}


def one_case(a):
    torch.cuda.set_device(0)
    # Dispatch threshold (drives the 3D/2D gate) is the real serving value,
    # unchanged by the buffer fix. The buffer first dim (bufdim) is the value the
    # fix changes: default = broken 128//nkv; pass --bufdim 256 to test the fix.
    seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // a.nkv
    bufdim = a.bufdim if a.bufdim is not None else seq_threshold_3D
    inp = build_batched(a.hs, a.nkv, a.nq, a.ctx, a.seqs, a.ntok)
    info = {"nq": a.nq, "hs": a.hs, "nkv": a.nkv, "ctx": a.ctx, "seqs": a.seqs,
            "ntok": a.ntok, "num_query_heads": a.nkv * a.nq, "total_q": inp["total_q"],
            "segm_bufdim": bufdim, "seq_threshold_3D": seq_threshold_3D, "gpu": _uuid}

    # 2D reference (serving fallback path)
    ref, ref_grid = run_wrapper(inp, use_3d=False, bufdim=bufdim,
                                seq_threshold_3D=seq_threshold_3D)
    # 3D candidate (serving path with sized segm buffers)
    try:
        cand, cand_grid = run_wrapper(inp, use_3d=True, bufdim=bufdim,
                                      seq_threshold_3D=seq_threshold_3D)
    except Exception as e:
        info["status"] = "CRASH"
        info["error"] = f"{type(e).__name__}: {str(e)[:240]}"
        info["candidate_grid"] = None
        print(json.dumps(info))
        return

    info["candidate_grid"] = cand_grid
    info["ref_grid"] = ref_grid
    ran_3d = (cand_grid is not None) and (len(cand_grid) == 3)
    if not ran_3d:
        # 3D was not selected -> the wrapper ran 2D. A numerical "pass" here is
        # meaningless for certifying the 3D path.
        info["status"] = "FELL BACK"
        info["note"] = "3D path did not run (grid is 2D)"
    else:
        r = compare(ref, cand)
        info["max_abs"] = r["max_abs"]
        info["max_rel"] = (None if (not r["has_rel"]) or math.isnan(r["max_rel"]) else r["max_rel"])
        info["status"] = "PASS" if r["pass"] else "FAIL"
    print(json.dumps(info))


def _case(nq, seqs, ntok, ctx, bufdim=None):
    cmd = [sys.executable, __file__, "--nq", str(nq), "--hs", "256",
           "--nkv", "1", "--ctx", str(ctx), "--seqs", str(seqs),
           "--ntok", str(ntok)]
    if bufdim is not None:
        cmd += ["--bufdim", str(bufdim)]
    env = dict(os.environ)
    env["GATE_GPU"] = _gpu
    oob = (ntok == 2 and seqs in (65, 128))
    timeout = 45 if oob else 90
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        out = (p.stdout or "").strip().splitlines()
        rec = json.loads(out[-1]) if out else {}
        rec["exit_code"] = p.returncode
        if p.returncode != 0 and "status" not in rec:
            rec["status"] = "CRASH" if p.returncode in (1, 137) else "HANG"
            rec["error"] = (p.stderr or "")[-240:] or "no output (process died / killed by timeout)"
    except subprocess.TimeoutExpired:
        rec = {"nq": nq, "seqs": seqs, "ntok": ntok, "ctx": ctx,
               "status": "HANG", "error": f"process still alive at {timeout}s (OOB busy loop)"}
    return rec


def run_matrix(bufdim=None):
    rows = []
    # Phase 1: all SAFE cases (OOB ones deferred to phase 2 so a hang is isolated).
    for nq in (4, 6):
        for seqs in (1, 8, 64, 65, 128):
            for ntok in (1, 2):
                for ctx in (2048, 16384):
                    if ntok == 2 and seqs in (65, 128):
                        continue
                    rec = _case(nq, seqs, ntok, ctx, bufdim=bufdim)
                    rec["phase"] = "safe"
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
    # Phase 2: the OOB cases (crash or hang). Wait for GPU to clear between each.
    for nq in (4, 6):
        for seqs in (65, 128):
            for ctx in (2048, 16384):
                rec = _case(nq, seqs, 2, ctx, bufdim=bufdim)
                rec["phase"] = "oob"
                rows.append(rec)
                print(json.dumps(rec), flush=True)
                _wait_gpu_idle()
    return rows


def _wait_gpu_idle(max_s=60):
    """Poll the G2 busy percent until it drops to 0 (OOB hang recovery), bounded."""
    import time
    base = "/sys/bus/pci/devices/"
    dev = None
    import glob
    for d in glob.glob(base + "*/unique_id"):
        try:
            if "cbfe8a265e29739c" in open(d).read():
                dev = os.path.dirname(d)
        except OSError:
            pass
    if dev is None:
        return
    t0 = time.time()
    while time.time() - t0 < max_s:
        try:
            busy = int(open(dev + "/gpu_busy_percent").read().strip())
        except OSError:
            return
        if busy == 0:
            return
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--nq", type=int)
    ap.add_argument("--hs", type=int, default=256)
    ap.add_argument("--nkv", type=int, default=1)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--seqs", type=int)
    ap.add_argument("--ntok", type=int)
    ap.add_argument("--bufdim", type=int, default=None)
    a = ap.parse_args()
    if a.all:
        run_matrix(bufdim=a.bufdim)
        return
    if a.nq is None or a.seqs is None or a.ntok is None or a.ctx is None:
        ap.error("need --nq --ctx --seqs --ntok (or --all)")
    one_case(a)


if __name__ == "__main__":
    main()
