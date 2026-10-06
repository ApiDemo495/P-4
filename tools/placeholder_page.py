"""A stand-in for port 8000 while the Codespace is still provisioning.

Codespaces opens the forwarded port in the browser as soon as the container
exists - minutes before pip has finished and the engine can listen.  Until
Round W that tab showed a connection error and the user concluded "it is not
starting".  This tiny server (standard library only, so it runs on the image's
system python before the virtualenv exists) answers on the port with a page
that shows the autostart journal live, refreshes itself every 5 seconds, and
hands the port over the moment the real engine starts (``run.sh`` and the
autostart hook stop it by pid).

Every ``/api/...`` request answers 503, so the health probes used by the hooks
(``curl -f /api/health``) keep reporting "not up yet" instead of being fooled.

    python3 tools/placeholder_page.py --port 8000 --root /path/to/repo
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTED = time.time()


def _tail(path: str, lines: int = 30) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64_000))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Drosophila Trader - starting</title>
<style>
 body{{background:#0b0f14;color:#dfe7ef;font:15px/1.5 system-ui,sans-serif;margin:0;padding:32px}}
 h1{{font-size:22px;margin:0 0 6px}} .dim{{color:#8795a3}}
 pre{{background:#121923;border:1px solid #223042;border-radius:8px;padding:14px;overflow:auto;font-size:12.5px;max-height:40vh}}
 .bar{{height:6px;background:#1c2634;border-radius:3px;overflow:hidden;margin:18px 0}}
 .bar i{{display:block;height:100%;width:40%;background:linear-gradient(90deg,#2f7d5c,#7fe0b5);animation:s 1.6s infinite linear}}
 @keyframes s{{from{{transform:translateX(-100%)}}to{{transform:translateX(260%)}}}}
</style></head><body>
<h1>Drosophila Trader v2.0 is setting itself up</h1>
<div class="dim">nothing to do - this page refreshes every 5 s and becomes the dashboard by itself · {elapsed}s so far · step: <b>{step}</b></div>
<div class="bar"><i></i></div>
<h3>autostart journal</h3><pre>{journal}</pre>
<h3>provisioning log</h3><pre>{setup}</pre>
<p class="dim">First start downloads ~150 MB of Python packages (once).</p>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    root = ROOT

    def log_message(self, *_args) -> None:  # quiet
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/") or self.path.startswith("/ws"):
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "5")
            self.end_headers()
            self.wfile.write(b'{"ready":false,"placeholder":true,"detail":"engine still provisioning"}')
            return
        journal = _tail(os.path.join(self.root, ".run", "logs", "autostart.journal"), 25)
        setup = ""
        for name in ("setup-3.log", "setup-2.log", "setup-1.log"):
            setup = _tail(os.path.join(self.root, ".run", "logs", name), 18)
            if setup:
                break
        if not setup:
            setup = _tail("/tmp/pip-install.log", 12) or "(waiting for the first log line)"
        last = journal.splitlines()[-1] if journal else "starting"
        step = last.split(" ", 3)[-1] if " " in last else last
        body = PAGE.format(
            elapsed=int(time.time() - STARTED),
            step=html.escape(step[:120]),
            journal=html.escape(journal or "(no entries yet)"),
            setup=html.escape(setup),
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_HEAD = do_GET


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    Handler.root = args.root
    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    except OSError as exc:
        print(f"placeholder: port {args.port} busy ({exc}) - not needed", file=sys.stderr)
        return 0
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
