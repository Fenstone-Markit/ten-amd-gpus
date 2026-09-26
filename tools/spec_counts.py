import re, sys, json, urllib.request
def snap(port):
    txt = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5).read().decode()
    d = {}
    for line in txt.splitlines():
        m = re.match(r'^(vllm:spec_decode_[a-z_]+?)(_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if m and "per_pos" not in m.group(1):
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(4))
    return d
if sys.argv[1] == "save":
    json.dump(snap(int(sys.argv[2])), open(sys.argv[3], "w"))
else:
    a, b = json.load(open(sys.argv[3])), snap(int(sys.argv[2]))
    delta = {k: b.get(k, 0) - a.get(k, 0) for k in b}
    drafts, drafted, accepted = delta.get("vllm:spec_decode_num_drafts", 0), delta.get("vllm:spec_decode_num_draft_tokens", 0), delta.get("vllm:spec_decode_num_accepted_tokens", 0)
    if drafted <= 0:
        print(f"  no speculative counters moved; counters seen: {sorted(b)}")
    else:
        print(f"  {sys.argv[4]}: {accepted:.0f} of {drafted:.0f} draft tokens accepted ({100 * accepted / drafted:.1f}%); "
              f"mean tokens per verify step {1 + accepted / max(drafts, 1):.2f}")
