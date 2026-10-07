"""Round AK: the neuPrint token test must use endpoints that still exist."""
from __future__ import annotations

import json

import httpx
import pytest

from backend.brain import health_check as hc


def _patched_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_token_test_never_calls_database_info(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.url.path == "/api/custom/custom":
            body = json.loads(request.content)
            assert body["dataset"] == "hemibrain:v1.2.1"
            return httpx.Response(200, json={"columns": ["dataset", "edited"], "data": [["hemibrain:v1.2.1", "x"]]})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    out = await hc.neuprint_test_token("a" * 64, "https://neuprint.janelia.org", "hemibrain:v1.2.1")
    assert out["valid"] is True, out
    assert all(path != "/api/databaseInfo" for _, path, _ in seen)
    assert seen[0][2] == "Bearer " + "a" * 64


@pytest.mark.asyncio
async def test_token_rejected_is_401_not_404(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad token"})

    _patched_client(monkeypatch, handler)
    out = await hc.neuprint_test_token("b" * 64)
    assert out["valid"] is False and "rejected" in out["error"]


@pytest.mark.asyncio
async def test_query_route_missing_falls_back_to_dbmeta(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/dbmeta/datasets":
            return httpx.Response(200, json={"hemibrain:v1.2.1": {}, "male-cns:v0.9": {}})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    out = await hc.neuprint_test_token("c" * 64)
    assert out["valid"] is True and "present" in out["detail"]
