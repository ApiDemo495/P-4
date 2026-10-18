"""Round BQ - the jevos decision agent (feder-cr/jev) as the primary AI agent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from backend.agents import jev_agent
from backend.agents.base import AgentStatus
from backend.agents.jev_agent import JevAgent, build_questions, build_state, combine, parse_sums
from backend.core import config as cfg

ROOT = Path(__file__).resolve().parents[2]

CONTEXT = {
    "asset": "BTC", "price": 67321.5, "timestamp": "2026-10-18T02:00:00Z",
    "formulas": {"TAI": 0.42, "MPL": 0.31, "FIH": 0.12, "RBB": 0.2, "AFPR": 0.05, "MCPE": 0.3,
                 "DGW": -0.2, "LCS": 0.1, "BAR": 0.3, "VSD": 0.0, "RAR": 0.1, "PRT": 0.2, "WDR": 0.1,
                 "ERC": 0.3, "LAI": 0.1, "ANA": 0.2, "NIV": 0.15, "SMD": 0.05, "CCSv2": 0.4, "KCAE": 0.6,
                 "DSKD": 0.1, "PIN": 0.12, "HSI": 0.2},
    "ccs_confidence": 0.7, "drg": 0.1, "win_rate": 0.55, "headline": "ETF inflows hit a record", "brain_status": "LIVE",
}


def _server(answers: dict, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready", "model": "jevos-v5", "fingerprint": "abc123def456"})
        if request.url.path == "/v1/systemone":
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"model": "jevos-v5", "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 0}})
        return httpx.Response(404, json={"detail": "Not Found"})
    return httpx.MockTransport(handler)


def test_state_and_questions_follow_jev_wire_format():
    state = build_state(CONTEXT)
    assert state["asset"] == "BTC/USDT" and "70 seconds" in state["rule_of_the_window"]
    assert "TAI +0.420 (up)" in state["formulas micro-structure"]
    assert "point up" in state["tally"]
    qs = build_questions()
    assert qs["side"]["type"] == "choice" and set(qs["side"]["criteria"]) == {"up", "down"}
    for qid, _sign, _names, _rule in jev_agent.RULES:
        assert qs[qid]["type"] == "noul" and qs[qid]["instructions"].startswith("Rule:")
    assert "70 seconds" in qs["continuation"]["instructions"]


def test_combine_orients_every_rule_toward_up():
    p_up, parts = combine({
        "continuation": {"type": "noul", "noul": 0.8},
        "exhaustion": {"type": "noul", "noul": 0.9},          # "yes" = down -> 0.1 up
        "side": {"type": "choice", "choice": "up", "probabilities": {"up": 0.7, "down": 0.3}, "confidence": 0.4},
    })
    assert parts["exhaustion"] == pytest.approx(0.1)
    assert p_up == pytest.approx(0.5 * 0.7 + 0.5 * (0.8 + 0.1) / 2)
    assert combine({}) == (0.5, {})


async def test_agent_decides_from_probabilities_not_text(tmp_path):
    seen: list = []
    answers = {
        "continuation": {"type": "noul", "noul": 0.81},
        "flow": {"type": "noul", "noul": 0.55},
        "exhaustion": {"type": "noul", "noul": 0.30},
        "news": {"type": "noul", "noul": 0.66},
        "brain": {"type": "noul", "noul": 0.72},
        "side": {"type": "choice", "choice": "up", "probabilities": {"up": 0.68, "down": 0.32}, "confidence": 0.36},
    }
    agent = JevAgent(cfg.SETTINGS, home=tmp_path, transport=_server(answers, seen))
    assert (await agent.probe())["ok"] and agent.ready and agent.model == "jevos-v5"
    result = await agent.decide(CONTEXT, 0.30)
    assert result.status is AgentStatus.ACTIVE and result.decision == "BUY"
    expected = 0.5 * 0.68 + 0.5 * (0.81 + 0.55 + 0.70 + 0.66 + 0.72) / 5
    assert result.confidence == pytest.approx(abs(2 * expected - 1), abs=1e-6)
    assert "P(up)" in result.reasoning and "option logits" in result.reasoning
    body = seen[0]
    assert body["model"] == "jev-latest" and "state" in body and "side" in body["questions"]
    health = agent.health().to_dict()
    assert health["status"] == "ACTIVE" and health["calls"] == 1 and health["release"] == "jevos-v5"
    test = await agent.test()
    assert test["valid"] and "jevos-v5" in test["detail"]
    await agent.shutdown()


async def test_agent_sells_on_low_probabilities_and_reports_errors(tmp_path):
    answers = {"continuation": {"type": "noul", "noul": 0.2}, "side": {"type": "choice", "choice": "down",
               "probabilities": {"up": 0.3, "down": 0.7}, "confidence": 0.4}}
    agent = JevAgent(cfg.SETTINGS, home=tmp_path, transport=_server(answers, []))
    await agent.probe()
    result = await agent.decide(CONTEXT, 0.30)
    assert result.decision == "SELL" and result.confidence > 0.02

    def failing(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready", "model": "jevos-v5"})
        return httpx.Response(422, json={"detail": [{"msg": "Value error, State must not be empty"}]})
    broken = JevAgent(cfg.SETTINGS, home=tmp_path, transport=httpx.MockTransport(failing))
    await broken.probe()
    result = await broken.decide(CONTEXT, 0.30)
    assert result.status is AgentStatus.ERROR and "422" in result.error and result.decision is None


async def test_agent_not_ready_is_an_honest_status(tmp_path, monkeypatch):
    def down(request):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setenv("JEV_AUTOINSTALL", "0")
    agent = JevAgent(cfg.SETTINGS, home=tmp_path, transport=httpx.MockTransport(down))
    assert not agent.installed and not agent.configured
    result = await agent.decide(CONTEXT, 0.30)
    assert result.status is AgentStatus.DISABLED and result.decision is None
    monkeypatch.setenv("JEV_AUTOINSTALL", "1")
    agent = JevAgent(cfg.SETTINGS, home=tmp_path, transport=httpx.MockTransport(down))
    assert agent.configured
    result = await agent.decide(CONTEXT, 0.30)
    assert result.status is AgentStatus.INACTIVE and "not installed yet" in result.error
    assert agent.health().status is AgentStatus.LOADING


def test_release_checksums_are_parsed_and_verified(tmp_path):
    blob = tmp_path / "jev-linux-x64.tar.gz"
    blob.write_bytes(b"not really an archive")
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    sums = parse_sums(f"{digest}  jev-linux-x64.tar.gz\n{'0' * 64}  jevos-v5-openvino-int8.zip\n")
    assert sums["jev-linux-x64.tar.gz"] == digest and jev_agent.sha256_of(blob) == digest
    assert jev_agent.platform_archive() in (None, "jev-linux-x64.tar.gz", "jev-macos-arm64.tar.gz", "jev-windows-x64.zip")
    assert jev_agent.find_binary(tmp_path) is None and jev_agent.find_model_dir(tmp_path) is None


def test_jev_is_wired_as_the_primary_agent():
    from backend.agents.orchestrator import AgentOrchestrator
    from backend.core import lock_weights

    orch = AgentOrchestrator(cfg.SETTINGS, local_stub=True)
    weights = orch.weights()
    assert set(orch.agents) == {"jev", "gemini", "local"}
    assert weights["jev"] > weights["gemini"] and weights["jev"] > weights["local"]
    assert weights["drosophila"] + weights["jev"] + weights["gemini"] + weights["local"] == pytest.approx(1.0)
    assert lock_weights.BACKERS["jev"] == ("agent:jev",)
    for path, needle in (("backend/agents/fusion.py", '("jev", "gemini", "local")'),
                         ("backend/core/prediction.py", '("jev", "gemini", "local")'),
                         ("backend/core/cycle_manager.py", "self.agents.jev.start()"),
                         ("backend/api/routes_agents.py", 'manager.agents.jev.test()'),
                         ("backend/web/app.js", 'key: "jev"')):
        assert needle in (ROOT / path).read_text(encoding="utf-8"), path


def test_installer_unpacks_release_layout_and_server_child_lifecycle(tmp_path):
    import io
    import tarfile
    import zipfile

    from backend.agents.jev_agent import JevInstaller, JevServer, find_binary, find_model_dir

    home = tmp_path / "jev"
    home.mkdir()
    # jev-linux-x64.tar.gz: jev/jev (a stand-in script that sleeps like a server)
    tar_path = home / "jev-linux-x64.tar.gz"
    script = b"#!/bin/sh\nsleep 30\n"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("jev/jev")
        info.size = len(script)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(script))
    # jevos-v5-openvino-int8.zip: model/{model.json, openvino_model.xml, openvino_model.bin, tokenizer.gguf}
    zip_path = home / "jevos-v5-openvino-int8.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("model/model.json", json.dumps({"name": "jevos-v5"}))
        for name in ("openvino_model.xml", "openvino_model.bin", "tokenizer.gguf"):
            zf.writestr(f"model/{name}", b"x")
        zf.writestr("../escape.txt", b"nope")          # must be skipped
    installer = JevInstaller(home)
    installer._unpack(tar_path)
    installer._unpack(zip_path)
    binary, model_dir = find_binary(home), find_model_dir(home)
    assert binary is not None and binary.stat().st_mode & 0o111
    assert model_dir is not None and (model_dir / "model.json").is_file()
    assert not (tmp_path / "escape.txt").exists()

    server = JevServer(home, 18017, log_path=tmp_path / "jev.log")
    assert server.start() and server.alive and server.launches == 1
    assert server.start() and server.launches == 1               # idempotent while alive
    server.stop()
    assert not server.alive
