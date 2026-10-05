"""Round AB - connectivity probe: find the live internet the moment the app opens.

GitHub Codespaces have full egress, but not every exchange answers from the
US regions they run in (Binance answers 451 there).  Instead of discovering
that by failing a WebSocket three times over 45 seconds, the hub probes every
endpoint it could use **in parallel, once, at start-up** (4 s budget) and
starts with whatever is reachable.  The result is published on
``/api/health.connectivity`` and in the header chip, so "offline" always comes
with the HTTP status that says why.
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

log = logging.getLogger("drosophila.connectivity")

#: name -> (url, what it unlocks)
PROBES: dict[str, tuple[str, str]] = {
    "binance_vision": ("https://data-api.binance.vision/api/v3/ping", "Binance market-data mirror (WS tape + book)"),
    "binance": ("https://api.binance.com/api/v3/ping", "Binance (WS tape + book)"),
    "kraken": ("https://api.kraken.com/0/public/Time", "Kraken (WS tape + book, REST tape + book)"),
    "coingecko": ("https://api.coingecko.com/api/v3/ping", "CoinGecko (10 s prices)"),
    "news_rss": ("https://feeds.bbci.co.uk/news/world/rss.xml", "world news wire (RSS)"),
    "cryptopanic": ("https://cryptopanic.com/", "CryptoPanic (crypto news)"),
}
TIMEOUT_SECONDS = 4.0


async def _one(client: httpx.AsyncClient, name: str, url: str, purpose: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.get(url)
        ms = (time.perf_counter() - t0) * 1000.0
        ok = 200 <= r.status_code < 400
        return {"ok": ok, "status": r.status_code, "ms": round(ms, 1), "url": url, "purpose": purpose,
                "error": "" if ok else f"HTTP {r.status_code}" + (" geo-blocked" if r.status_code == 451 else "")}
    except Exception as exc:  # noqa: BLE001
        ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "status": None, "ms": round(ms, 1), "url": url, "purpose": purpose,
                "error": f"{type(exc).__name__}: {str(exc)[:100]}".strip(": ")}


async def probe(names: tuple[str, ...] | None = None, timeout: float = TIMEOUT_SECONDS) -> dict:
    """Probe every endpoint in parallel; never raises."""
    chosen = {k: v for k, v in PROBES.items() if names is None or k in names}
    out: dict = {"probed_at": time.time(), "timeout_s": timeout, "hosts": {}}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                     headers={"user-agent": "drosophila-trader/2.0"}) as client:
            results = await asyncio.gather(*(_one(client, n, u, p) for n, (u, p) in chosen.items()))
    except Exception as exc:  # noqa: BLE001
        results = [{"ok": False, "status": None, "ms": 0.0, "url": u, "purpose": p, "error": str(exc)[:100]}
                   for (u, p) in chosen.values()]
    for name, result in zip(chosen, results):
        out["hosts"][name] = result
    hosts = out["hosts"]
    out["internet"] = any(h["ok"] for h in hosts.values())
    out["binance_reachable"] = bool(hosts.get("binance_vision", {}).get("ok") or hosts.get("binance", {}).get("ok"))
    out["kraken_reachable"] = bool(hosts.get("kraken", {}).get("ok"))
    out["coingecko_reachable"] = bool(hosts.get("coingecko", {}).get("ok"))
    out["summary"] = summarize(out)
    log.info("connectivity: %s", out["summary"])
    return out


def summarize(report: dict) -> str:
    hosts = report.get("hosts") or {}
    if not hosts:
        return "not probed"
    if not report.get("internet"):
        return "no internet egress: " + "; ".join(f"{k} {v.get('error')}" for k, v in hosts.items())[:200]
    good = [k for k, v in hosts.items() if v.get("ok")]
    bad = [f"{k} ({v.get('error')})" for k, v in hosts.items() if not v.get("ok")]
    return f"reachable: {', '.join(good)}" + (f" · blocked: {', '.join(bad)}" if bad else "")


async def _ws_probe(url: str, subscribe: str | None, timeout: float = 8.0) -> dict:
    """Open a real WebSocket, optionally subscribe, wait for the first frame."""
    import websockets

    t0 = time.perf_counter()
    try:
        async with websockets.connect(url, open_timeout=timeout, close_timeout=2) as ws:
            if subscribe:
                await ws.send(subscribe)
            frame = await asyncio.wait_for(ws.recv(), timeout=timeout)
            ms = (time.perf_counter() - t0) * 1000.0
            text = frame if isinstance(frame, str) else frame.decode("utf-8", "replace")
            return {"ok": True, "ms": round(ms, 1), "url": url, "first_frame": text[:160], "error": ""}
    except Exception as exc:  # noqa: BLE001
        ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "ms": round(ms, 1), "url": url, "first_frame": "",
                "error": f"{type(exc).__name__}: {str(exc)[:120]}".strip(": ")}


async def diagnose(settings=None) -> dict:
    """Round AE - everything the engine needs from the network, tested live:
    HTTP probes, a real WebSocket handshake + subscribe on Kraken and Binance,
    and one RSS fetch.  Served at ``/api/feeds/diagnose`` so a single
    ``curl`` shows WHY a Codespace reads &b offline."""
    import json

    http_task = probe()
    kraken_sub = json.dumps({"method": "subscribe", "params": {"channel": "trade", "symbol": ["BTC/USD"], "snapshot": True}})
    ws_tasks = {
        "kraken_ws": _ws_probe("wss://ws.kraken.com/v2", kraken_sub),
        "binance_ws": _ws_probe("wss://data-stream.binance.vision/ws/btcusdt@aggTrade", None),
    }

    async def rss() -> dict:
        t0 = time.perf_counter()
        try:
            from backend.news import rss_source

            feeds = list(getattr(settings, "rss_feeds", None) or []) or ["https://feeds.bbci.co.uk/news/world/rss.xml"]
            items = await rss_source.fetch_feed(feeds[0], 5)
            return {"ok": bool(items), "ms": round((time.perf_counter() - t0) * 1000.0, 1), "url": feeds[0],
                    "items": len(items), "sample": [getattr(i, "headline", str(i))[:100] for i in list(items)[:3]],
                    "error": "" if items else "feed parsed but empty"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "ms": round((time.perf_counter() - t0) * 1000.0, 1), "url": "", "items": 0,
                    "sample": [], "error": f"{type(exc).__name__}: {str(exc)[:120]}"}

    http, kraken_ws, binance_ws, rss_result = await asyncio.gather(http_task, ws_tasks["kraken_ws"], ws_tasks["binance_ws"], rss())
    out = {"http": http, "websocket": {"kraken_ws": kraken_ws, "binance_ws": binance_ws}, "rss": rss_result}
    verdict = []
    if not http.get("internet"):
        verdict.append("no HTTP egress at all - the engine cannot reach any market or news host")
    if kraken_ws["ok"] or binance_ws["ok"]:
        verdict.append("a WebSocket tape is reachable: " + ", ".join(k for k, v in (("kraken", kraken_ws), ("binance", binance_ws)) if v["ok"]))
    else:
        verdict.append("no WebSocket egress - the engine must run on Kraken REST / CoinGecko polling")
    verdict.append("news RSS " + ("ok" if rss_result["ok"] else "FAILED: " + rss_result["error"]))
    out["verdict"] = "; ".join(verdict)
    return out
