# DROSOPHILA TRADER v2.0 — Flutter client

The mobile/desktop client for the engine in `../backend`. It is a pure view:
the signal it displays is computed and locked by the backend, and there is no
code path in this app that can alter it mid-cycle.

## Run it

```bash
cd frontend
flutter pub get
flutter run -d chrome --dart-define=API_BASE=http://localhost:8000
# or a device/emulator
flutter run --dart-define=API_BASE=http://192.168.1.20:8000
```

The engine must be running first:

```bash
PYTHONPATH=.. python -m backend.api.main       # from the repo root
```

`API_BASE` defaults to `http://localhost:8000`; the WebSocket URL is derived from
it (`http` → `ws`, `https` → `wss`, path `/ws/signals`).

## Structure

```
lib/
  main.dart                     app entry + provider wiring
  theme.dart                    dark palette, Panel, SignedBar, StatRow
  models/signal.dart            FrozenSignal, outcomes, formulas, news
  services/api_client.dart      REST mirror (+ --dart-define=API_BASE)
  services/signal_socket.dart   /ws/signals with exponential-backoff reconnect
  state/app_state.dart          ChangeNotifier: clock, signal, panels
  screens/dashboard_screen.dart Signal tab: timer, hedge, agents, news, history
  screens/formula_explorer_screen.dart   all 22 formulas in 8 categories
  screens/settings_screen.dart  key tests, local model, brain, news
  widgets/cycle_timer.dart      countdown + ⏳/🔒/⚡ lock state
  widgets/signal_panel.dart     FROZEN signal panel + HOLD box (Section 10.2)
  widgets/hedge_dashboard.dart  HSI/HRDD/SHRP/GCDV + brain read-out
  widgets/news_card.dart        headline, tier, poll age, NIV/SMD
  widgets/agent_list.dart       agents, statuses, weights + EmergencyOverlay
```

## Behaviour notes

* **⏳ / 🔒 / ⚡** — the countdown widget renders the lock state straight from the
  protocol; while `COMPUTING` the panel deliberately has nothing to show, because
  the backend raises `SignalNotReady` inside that window.
* **Rule 2** — the Formula Explorer refreshes on `FORMULA_UPDATE` every 15 s
  (0.75 s at `TIME_SCALE=20`) and states in a banner that the signal stays locked.
* **Queued asset switch** — tapping BTC/PAXG sends the action and shows
  "Switching to … at the next cycle boundary"; the engine applies it at `t = 60`.
* **Reconnect** — the socket backs off 1 s → 2 s → 4 s → 15 s and the server's
  `HELLO` frame restores the whole UI state, so a dropped connection never leaves
  a stale signal on screen.

## Tests

```bash
flutter test
```
