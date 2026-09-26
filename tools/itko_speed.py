"""itko_speed.py: the brain's real working speed, read from its own counters while Itko works.
usage: python3 itko_speed.py PORT SECONDS   (samples every 0.5 s; prints a summary at the end or on Ctrl-C)
Headline: decode speed of requests that finished in the window, from vLLM's own per-request timings
(generated tokens / decode seconds): exact, whatever the sampling. Cross-check: a sampled busy-time rate,
which reads a little low because partly busy intervals count as busy."""
import re, sys, time, urllib.request
port, dur = int(sys.argv[1]), float(sys.argv[2])
WANT = ("vllm:generation_tokens", "vllm:prompt_tokens", "vllm:num_requests_running", "vllm:request_prompt_tokens_sum",
        "vllm:request_prompt_tokens_count", "vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens", "vllm:request_decode_time_seconds_sum",
        "vllm:request_generation_tokens_sum", "vllm:request_prefill_time_seconds_sum")
def snap():
    txt = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5).read().decode()
    d = {}
    for line in txt.splitlines():
        m = re.match(r'^(vllm:[a-z_]+?)(_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if m and m.group(1) in WANT:
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(4))
    return d
first = prev = snap(); t0 = tp = time.time(); busy = busy_tok = 0.0
try:
    while time.time() - t0 < dur:
        time.sleep(0.5)
        cur = snap(); now = time.time()
        dtok = cur.get("vllm:generation_tokens", 0) - prev.get("vllm:generation_tokens", 0)
        if cur.get("vllm:num_requests_running", 0) >= 1 and dtok > 0:
            busy += now - tp; busy_tok += dtok
        prev, tp = cur, now
except KeyboardInterrupt:
    pass
last = prev; wall = tp - t0
d = lambda k: last.get(k, 0) - first.get(k, 0)
reqs = d("vllm:request_prompt_tokens_count")
dec_s, dec_tok = d("vllm:request_decode_time_seconds_sum"), d("vllm:request_generation_tokens_sum")
if dec_s > 0:
    print(f"DECODE SPEED, finished requests: {dec_tok / dec_s:.1f} tok/s ({dec_tok:.0f} tokens over {dec_s:.0f} s of decode; exact, from vLLM's per-request timings)")
else:
    print("no request finished in this window, so no exact decode speed; see the sampled rate below")
if reqs and d("vllm:request_prefill_time_seconds_sum") > 0:
    print(f"prefill per request: {d('vllm:request_prefill_time_seconds_sum') / reqs:.2f} s on average (the wait before each answer starts)")
print(f"window {wall / 60:.1f} min; busy generating {busy / 60:.1f} min ({100 * busy / max(wall, 1e-9):.0f}%)")
if busy_tok:
    print(f"cross-check, sampled: {busy_tok / busy:.1f} tok/s while busy (reads a little low: partly busy intervals count as busy)")
else:
    print("no generation seen in this window (was Itko working?)")
print(f"prompt tokens processed {d('vllm:prompt_tokens'):.0f}; requests finished {reqs:.0f}"
      + (f"; average prompt {d('vllm:request_prompt_tokens_sum') / reqs:.0f} tokens" if reqs else ""))
dr, acc, steps = d("vllm:spec_decode_num_draft_tokens"), d("vllm:spec_decode_num_accepted_tokens"), d("vllm:spec_decode_num_drafts")
if dr:
    print(f"speculation: {100 * acc / dr:.1f}% of draft tokens accepted; {1 + acc / max(steps, 1):.2f} tokens per step")
