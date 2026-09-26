import json, sys, time, urllib.request
port = int(sys.argv[1]); base = f"http://localhost:{port}"
for ctx in (16000, 60000):
    for run in (1, 2):
        p = "The quick brown fox jumps over the lazy dog. " * (ctx // 10)
        body = json.dumps({"model": "brain", "prompt": p, "max_tokens": 64, "min_tokens": 64, "temperature": 0,
                           "ignore_eos": True, "stream": True, "logprobs": 1}).encode()
        req = urllib.request.Request(base + "/v1/completions", body, {"Content-Type": "application/json"})
        t0 = time.time(); t_first = t_last = None; first_tokens = total = chunks = 0
        for line in urllib.request.urlopen(req, timeout=600):
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            ch = json.loads(line[6:])["choices"][0]
            n = len((ch.get("logprobs") or {}).get("tokens") or [])
            if n == 0:
                continue
            now = time.time()
            if t_first is None:
                t_first, first_tokens = now, n
            t_last = now; total += n; chunks += 1
        if chunks == 0:
            print(f"ctx~{ctx:>6} run {run}: no tokens streamed back; the server did not answer as expected", flush=True)
            continue
        after = total - first_tokens
        ms = (t_last - t_first) * 1000 / after if after > 0 else float("nan")
        print(f"ctx~{ctx:>6} run {run}: {total} tokens in {chunks} chunks ({total / chunks:.2f} per chunk), "
              f"first token {t_first - t0:5.2f}s, decode {ms:5.1f} ms/token = {1000 / ms:5.1f} tok/s", flush=True)
