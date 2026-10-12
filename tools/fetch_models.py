#!/usr/bin/env python3
"""Download every enabled model in models.json into a directory (Round AV).

Used by the dev-container image build so a new Codespace already contains the
weights, and usable by hand:  python tools/fetch_models.py --dest backend/models
Idempotent: files that exist with a sane size are kept.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.model_fetch import normalise_url  # noqa: E402

MIN_BYTES = 100 * 1024 * 1024


def manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [m for m in data.get("models", []) if m.get("enabled", True) and m.get("url")]


def fetch(url: str, dest: Path, quiet: bool = False) -> Path:
    name = Path(url.split("?", 1)[0]).name
    target = dest / name
    if target.exists() and target.stat().st_size >= MIN_BYTES:
        if not quiet:
            print(f"keep  {name} ({target.stat().st_size / 1e9:.2f} GB)")
        return target
    dest.mkdir(parents=True, exist_ok=True)
    part = dest / (name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "drosophila-trader/2.0"})
    for ctx in (None, ssl.create_default_context()):
        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp, part.open("wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done, t0 = 0, time.time()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if not quiet and total and time.time() - t0 > 5:
                        print(f"  {name}: {100 * done / total:.0f}%", flush=True)
                        t0 = time.time()
            break
        except ssl.SSLError:
            continue
    part.replace(target)
    if not quiet:
        print(f"done  {name} ({target.stat().st_size / 1e9:.2f} GB)")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "backend" / "models"))
    ap.add_argument("--manifest", default=str(ROOT / "models.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    entries = manifest(Path(args.manifest))
    if not entries:
        print("models.json lists no enabled models")
        return 0
    for entry in entries:
        fetch(normalise_url(entry["url"]), Path(args.dest), quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
