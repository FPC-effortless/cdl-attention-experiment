"""Run a private, local browser tester for a COG Tiny Shakespeare checkpoint.

Example:
    python serve.py --checkpoint runs/tiny10m_seed20261122/best.pt
Then open http://127.0.0.1:8765/ on the same computer.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading

from sample import generate, load_model


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>COG · Tiny Shakespeare</title>
<style>
:root{color-scheme:dark;--bg:#111b18;--panel:#1b2722;--line:#38483f;--text:#edf2e9;--muted:#b4c2b5;--accent:#d5e5a6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
main{max-width:1000px;margin:auto;padding:32px 24px 48px}
.eyebrow{color:var(--accent);letter-spacing:.16em;text-transform:uppercase;font-size:12px;font-weight:700;margin:0 0 10px}
h1{font-size:clamp(28px,5vw,42px);font-weight:650;letter-spacing:-.03em;margin:0 0 10px}
.subtitle{max-width:750px;color:var(--muted);margin:0 0 20px}
.meta{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:26px}
.tag{border:1px solid var(--line);border-radius:30px;padding:5px 11px;color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;min-width:0}
h2{font-size:17px;margin:0 0 18px;font-weight:600}
label{display:block;font-size:13px;font-weight:600;margin-bottom:7px}
textarea,input{width:100%;background:#121d17;border:1px solid #53654f;border-radius:8px;padding:11px 12px;color:var(--text);font:inherit}
textarea{resize:vertical;min-height:160px;line-height:1.55}
textarea:focus,input:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}
.help{font-size:12px;color:var(--muted);margin:6px 0 0}
.buttons{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:22px}
button{border:0;border-radius:8px;padding:11px 15px;background:var(--accent);color:#142010;font:inherit;font-size:14px;font-weight:650;cursor:pointer}
button.secondary{border:1px solid #627259;background:transparent;color:var(--text)}
button:disabled{opacity:.55;cursor:wait}
.output-heading{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:16px}
.output-heading h2{margin:0}
.output-heading button{padding:6px 11px;font-size:12px}
pre{font:17px/1.65 Georgia,"Times New Roman",serif;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;min-height:290px;max-height:650px;overflow:auto}
.placeholder{color:var(--muted);font-family:inherit;font-size:14px}
#status{min-height:24px;margin:14px 0 0;font-size:13px;color:var(--muted)}
#status.error{color:#ffb8ac}
.stats{font-size:12px;color:var(--muted);margin:16px 0 0;border-top:1px solid var(--line);padding-top:12px;min-height:30px}
footer{color:var(--muted);font-size:12px;margin-top:20px;max-width:760px}
@media(max-width:720px){main{padding:24px 16px}.grid{grid-template-columns:1fr}.panel{padding:18px}pre{min-height:180px}}
</style>
</head>
<body>
<main>
<p class="eyebrow">Woyengi · Research model</p>
<h1>COG Tiny Shakespeare</h1>
<p class="subtitle">Give the model a starting passage and explore its continuation. This is an experimental character language model, not an instruction-following chatbot.</p>
<div class="meta">
<span class="tag">Local inference</span>
<span class="tag" id="parameters">Loading checkpoint information…</span>
<span class="tag" id="checkpoint">Character tokenizer</span>
</div>
<div class="grid">
<section class="panel" aria-labelledby="input-heading">
<h2 id="input-heading">Start a passage</h2>
<form id="form">
<label for="prompt">Prompt</label>
<textarea id="prompt" name="prompt" spellcheck="false" maxlength="8192">ROMEO:
</textarea>
<p class="help">Try a character name and colon, or a line of dialogue. Use the corpus's English characters; unsupported symbols produce an explicit error.</p>
<div class="controls">
<div>
<label for="length">New characters</label>
<input id="length" type="number" min="1" max="2000" step="1" value="400" required>
<p class="help">Longer continuations take more time.</p>
</div>
<div>
<label for="temperature">Temperature</label>
<input id="temperature" type="number" min="0" max="2" step="0.05" value="0.8" required>
<p class="help">0 is greedy; higher values add variation.</p>
</div>
<div>
<label for="seed">Seed</label>
<input id="seed" type="number" min="0" max="2147483647" step="1" value="42" required>
<p class="help">Keep settings and seed to repeat a sample.</p>
</div>
</div>
<div class="buttons">
<button id="submit" type="submit">Generate continuation</button>
<button class="secondary" id="reset" type="button">Reset prompt</button>
</div>
<p id="status" role="status" aria-live="polite"></p>
</form>
</section>
<section class="panel" aria-labelledby="output-heading">
<div class="output-heading">
<h2 id="output-heading">Continuation</h2>
<button class="secondary" id="copy" type="button" disabled>Copy text</button>
</div>
<pre id="output" class="placeholder">Your prompt and the generated continuation will appear here.</pre>
<p class="stats" id="stats"></p>
</section>
</div>
<footer>The server runs on this computer at 127.0.0.1. Prompts are processed by the loaded checkpoint without an external model service. This small, single-corpus model may repeat training-like phrases, make mistakes, or lose coherence.</footer>
</main>
<script>
const form=document.getElementById("form"), submit=document.getElementById("submit");
const status=document.getElementById("status"), output=document.getElementById("output"), stats=document.getElementById("stats"), copy=document.getElementById("copy");
function showStatus(message,error=false){status.textContent=message;status.className=error?"error":"";}
fetch("/api/info").then(r=>r.json()).then(info=>{
  document.getElementById("parameters").textContent=info.parameters.toLocaleString()+" parameters";
  document.getElementById("checkpoint").textContent=info.vocab_size+" characters · step "+info.checkpoint_step;
}).catch(()=>{document.getElementById("parameters").textContent="Information unavailable";});
form.addEventListener("submit",async event=>{
  event.preventDefault();
  const request={prompt:document.getElementById("prompt").value,
    max_new_tokens:Number(document.getElementById("length").value),
    temperature:Number(document.getElementById("temperature").value),
    seed:Number(document.getElementById("seed").value)};
  submit.disabled=true;copy.disabled=true;
  showStatus("Generating on this computer…");
  try{
    const response=await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(request)});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||"Generation failed.");
    output.textContent=result.text;output.className="";
    stats.textContent=result.stats.generated_characters+" new characters · "+result.stats.seconds_including_prefill.toFixed(2)+" seconds including prompt · seed "+result.stats.seed;
    showStatus("Generation complete.");copy.disabled=false;
  }catch(error){showStatus(error.message||"Could not reach the local server.",true);}
  finally{submit.disabled=false;}
});
document.getElementById("reset").addEventListener("click",()=>{document.getElementById("prompt").value="ROMEO:\n";showStatus("");});
copy.addEventListener("click",async()=>{
  try{await navigator.clipboard.writeText(output.textContent);showStatus("Text copied.");}
  catch{showStatus("Clipboard access is unavailable. Select the output text to copy it.",true);}
});
</script>
</body>
</html>"""


def make_handler(model, tokenizer, metadata, device, port):
    lock = threading.Lock()
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {"http://" + host for host in allowed_hosts}
    info = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tokenizer.chars),
        "checkpoint_step": metadata.get("step", "unknown"),
        "device": device,
        "architecture_id": model.config.architecture_id,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "COGTinyLocal/1.0"

        def headers_for(self, code, mime, length):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()

        def json_response(self, code, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.headers_for(code, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def valid_local_request(self):
            if self.headers.get("Host", "") not in allowed_hosts:
                self.json_response(403, {"error": "Use the printed localhost address."})
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                self.json_response(403, {"error": "Cross-site requests are not accepted."})
                return False
            return True

        def do_GET(self):
            if not self.valid_local_request():
                return
            if self.path == "/":
                body = HTML.encode("utf-8")
                self.headers_for(200, "text/html; charset=utf-8", len(body))
                self.wfile.write(body)
            elif self.path == "/api/info":
                self.json_response(200, info)
            elif self.path == "/favicon.ico":
                self.headers_for(204, "image/x-icon", 0)
            else:
                self.json_response(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_local_request():
                return
            if self.path != "/api/generate":
                self.json_response(404, {"error": "Not found"})
                return
            if self.headers.get_content_type() != "application/json":
                self.json_response(415, {"error": "Expected application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 65536:
                    raise ValueError("Request must be at most 64 KiB")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")
                prompt = request.get("prompt", "\n")
                count = request.get("max_new_tokens", 400)
                temperature = request.get("temperature", 0.8)
                seed = request.get("seed", 42)
                if not isinstance(prompt, str) or len(prompt) > 8192:
                    raise ValueError("Prompt must be text of at most 8,192 characters")
                if type(count) is not int or not 1 <= count <= 2000:
                    raise ValueError("New-character count must be an integer from 1 to 2,000")
                if type(seed) is not int or not 0 <= seed <= 2147483647:
                    raise ValueError("Seed must be an integer from 0 to 2,147,483,647")
                if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
                    raise ValueError("Temperature must be a finite number from 0 to 2")
                tokenizer.encode(prompt or "\n")
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                self.json_response(400, {"error": str(exc)})
                return
            if not lock.acquire(blocking=False):
                self.json_response(429, {"error": "The model is already generating. Wait for that request to finish."})
                return
            try:
                text, stats = generate(
                    model, tokenizer, prompt, max_new_tokens=count,
                    temperature=float(temperature), seed=seed, device=device,
                )
                self.json_response(200, {"text": text, "stats": stats})
            except (ValueError, RuntimeError) as exc:
                self.json_response(500, {"error": str(exc)})
            finally:
                lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.threads < 1:
        parser.error("--threads must be positive")
    model, tokenizer, metadata = load_model(args.checkpoint, args.device, args.threads)
    handler = make_handler(model, tokenizer, metadata, args.device, args.port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters.", flush=True)
    print(f"Open http://127.0.0.1:{args.port}/ on this computer. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

