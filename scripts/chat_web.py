#!/usr/bin/env python3
"""Minimal local web chat UI for SFT checkpoints (stdlib HTTP, no new deps).

Serves a single-page chat at http://127.0.0.1:7860 wired to the model via
the same generation loop as chat_repl.py (chatml template, <|end|> stop,
repetition penalty).

  uv run python scripts/chat_web.py
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sentencepiece as spm
import torch

from scripts.chat_repl import generate_reply
from scripts.infer import load_model

PAGE = """<!doctype html><meta charset=utf-8><title>gpt-25m chat</title>
<style>
body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
#log{border:1px solid #333;border-radius:8px;padding:1rem;min-height:300px;margin-bottom:1rem;white-space:pre-wrap}
.u{color:#8ab4f8}.b{color:#81c995}.meta{color:#777;font-size:.85em}
form{display:flex;gap:.5rem}input{flex:1;padding:.6rem;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#eee}
button{padding:.6rem 1rem;border-radius:6px;border:0;background:#8ab4f8;color:#111;font-weight:600}
</style>
<h2>gpt-25m-chat <span class=meta>(24.9M params &middot; expect confident nonsense)</span></h2>
<div id=log></div>
<form onsubmit="send(event)"><input id=msg autofocus placeholder="say something"><button>send</button></form>
<p class=meta><a style=color:#777 href=# onclick="hist=[];log.textContent='';return false">reset history</a></p>
<script>
let hist=[];const log=document.getElementById('log'),msg=document.getElementById('msg');
async function send(e){e.preventDefault();const t=msg.value.trim();if(!t)return;msg.value='';
log.textContent+=`you> ${t}\n`;log.textContent+=`bot> ...`;
const r=await fetch('/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text:t,history:hist})});
const d=await r.json();log.textContent=log.textContent.replace(/bot> \\.\\.\\.$/, `bot> ${d.reply}\n\n`);
hist.push([t,d.reply]);window.scrollTo(0,document.body.scrollHeight);}
</script>"""


class Handler(BaseHTTPRequestHandler):
    model = None
    sp = None
    device = "cpu"

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        text = str(body.get("text", ""))[:2000]
        history = [(str(u)[:2000], str(a)[:2000]) for u, a in body.get("history", [])][-3:]
        reply = generate_reply(self.model, self.sp, history, text, self.device)
        payload = json.dumps({"reply": reply}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/sft/25m-dolly-rehearsal-20260730/runs/sft/25m-dolly-rehearsal/final",
    )
    parser.add_argument("--model_config", default="configs/model_25m.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/spm.model")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"loading model on {args.device} ...")
    Handler.model, _ = load_model(args.checkpoint, args.model_config, args.device)
    Handler.sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    Handler.device = args.device
    print(f"chat UI: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
