"""End-to-end smoke test: run the real cycle manager for N seconds.

    python -m backend.tests.smoke --seconds 25 --scale 20

Prints every broadcast message and a final summary, so a full cycle (formulas,
agents, fusion, lock, outcome tracking) can be verified without a UI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time


def _configure(scale: float, stub: bool) -> None:
    os.environ.setdefault("TIME_SCALE", str(scale))
    os.environ.setdefault("MARKET_DATA_MODE", "simulator")
    os.environ.setdefault("LOG_LEVEL", "info")
    if stub:
        os.environ["LOCAL_AGENT_STUB"] = "1"


async def run(seconds: float, scale: float, stub: bool, quiet: bool) -> dict:
    _configure(scale, stub)

    from backend.core import config as cfg
    from backend.core.cycle_manager import CycleManager

    settings = cfg.Settings()
    manager = CycleManager(settings, local_stub=stub)
    await manager.start()
    manager.mark_started()

    queue = manager.subscribe()
    seen: dict[str, int] = {}
    signals: list[dict] = []
    deadline = time.time() + seconds

    try:
        while time.time() < deadline:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            kind = message.get("type", "?")
            seen[kind] = seen.get(kind, 0) + 1
            if kind == "SIGNAL":
                signals.append(message["data"])
                if not quiet:
                    data = message["data"]
                    print(
                        f"[{data['timestamp']}] cycle {data['cycle_number']:>3} "
                        f"{data['asset']:<4} {data['signal']:<4} "
                        f"conf={data['confidence']:.2f} ccs={data['ccs_value']:+.2f} "
                        f"ccs_conf={data['ccs_confidence']:.2f} price={data['price']:.2f} "
                        f"formulas={data['total_ms']:.1f}ms"
                    )
            elif kind in ("EMERGENCY_OVERRIDE", "OUTCOME", "CYCLE_START") and not quiet:
                if kind == "EMERGENCY_OVERRIDE":
                    print(f"  !! EMERGENCY: {message['data']['headline']}")
                elif kind == "OUTCOME":
                    print(
                        f"  outcome cycle {message['data']['cycle_number']}: "
                        f"{message['data']['signal']} -> {message['data']['pnl_bps']:+.1f} bps "
                        f"(win_rate={message['data']['win_rate']:.0%})"
                    )
    finally:
        summary = {
            "messages": seen,
            "signals": len(signals),
            "exit_status": manager.status(),
            "health": manager.health(),
            "last_signal": signals[-1] if signals else None,
        }
        await manager.stop()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--stub", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    summary = asyncio.run(run(args.seconds, args.scale, args.stub, args.quiet))
    print("\n--- summary ---")
    print("messages:", summary["messages"])
    print("signals :", summary["signals"])
    print("degradation:", summary["exit_status"]["degradation_label"])
    print("brain:", summary["health"]["brain"]["status"], summary["health"]["brain"]["message"])
    if summary["last_signal"]:
        last = summary["last_signal"]
        print("\nlast signal formulas:")
        for name, value in last["formulas"].items():
            if not name.startswith("_"):
                print(f"   {name:<6} {value:+.4f}")
        print("\nagents:")
        for name, payload in last["agents"].items():
            print(f"   {name:<10} {payload['status']:<12} {str(payload['decision']):<5} {payload['error'][:60]}")
        print("\nreasoning:", last["reasoning"])
        print("hold_warning:", json.dumps(last.get("hold_warning"))[:120])


if __name__ == "__main__":
    main()
