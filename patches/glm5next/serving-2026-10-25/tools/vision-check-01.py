# vision-check-01.py (run inside the GLM container, which has PIL): draw codes on images, ask GLM to read them back.
import base64, io, json, time, urllib.request
from PIL import Image, ImageDraw, ImageFont
def img(text, w=640, h=360, bg="white", fg="black"):
    im = Image.new("RGB", (w, h), bg); d = ImageDraw.Draw(im); f = ImageFont.load_default(size=72)
    d.text((40, 120), text, fill=fg, font=f); d.rectangle([10, 10, w - 10, h - 10], outline="red", width=6)
    b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
tests = [("K4821", "What code is written in the image? Answer with the code only."),
         ("GATE 7 = K309", "Read the text in the image exactly."),
         ("K5566", "What colour is the border of this image? One word.")]
for text, q in tests:
    body = {"model": "glm", "temperature": 0, "max_tokens": 200, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img(text)}}, {"type": "text", "text": q}]}]}
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read())
        c = r["choices"][0]["message"].get("content") or ""
        print(f"{q[:40]!r} on image {text!r}: {c.strip()[:80]!r} ({r['usage']['prompt_tokens']} prompt tokens, {time.time() - t:.1f} s)", flush=True)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:300], flush=True)
