import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/signal.dart';
import '../services/api_client.dart';
import '../services/signal_socket.dart';

/// Single source of truth for the Flutter client.
///
/// It renders exactly what the backend locked. The client never derives a
/// signal of its own, and it cannot show a different one mid-cycle because the
/// only `SIGNAL` event it accepts is the locked payload.
class AppState extends ChangeNotifier {
  AppState({ApiClient? api, SignalSocket? socket})
      : api = api ?? ApiClient(),
        socket = socket ?? SignalSocket(ApiClient.resolvedBase);

  final ApiClient api;
  final SignalSocket socket;

  // --- connection / clock --------------------------------------------------
  bool connected = false;
  double cyclePeriodSeconds = 60;
  double secondsRemaining = 60;
  int cycleNumber = 0;
  int degradationLevel = 1;
  String degradationLabel = '';
  bool ntpSynced = true;

  // --- signal --------------------------------------------------------------
  String asset = 'BTC';
  String? pendingAsset;
  String lockState = 'COMPUTING';
  String lockIcon = '\u{23F3}';
  FrozenSignal? signal;
  HoldWarning? holdWarning;

  /// The prediction block: side, freshness, 1:1 levels and reasoning.
  Prediction prediction = Prediction.none;

  /// When [prediction] was received (server-clock ms), so its age ticks on the
  /// same clock as the countdown.
  int? predictionReceivedAtMs;

  /// The prediction's age right now, advanced on the *server* clock - the same
  /// one the countdown runs on - so the chip and the digits cannot disagree.
  double get predictionAgeSeconds {
    final base = prediction.ageSeconds;
    if (base == null) return 0.0;
    final at = predictionReceivedAtMs;
    if (at == null) return base;
    return base + (serverNowMs() - at) / 1000.0;
  }

  bool get predictionStale =>
      prediction.ageSeconds != null && predictionAgeSeconds > prediction.maxAgeSeconds;
  Map<String, double> liveFormulas = {};
  Map<String, double> lastLockedFormulas = {};
  Map<String, double> weights = const {};

  // --- panels --------------------------------------------------------------
  List<FormulaCategory> catalog = const [];
  NewsSnapshot? news;
  List<SignalOutcome> outcomes = const [];
  List<Map<String, dynamic>> history = const [];
  List<Map<String, dynamic>> criticalEvents = const [];
  Map<String, dynamic> agentStatus = const {};
  Map<String, dynamic> brainStatus = const {};
  Map<String, dynamic> timings = const {};
  Map<String, dynamic> timingsMap = const {};
  double totalFormulaMs = 0.0;
  Map<String, dynamic> buyAccuracy = const {};
  Map<String, dynamic> sellAccuracy = const {};
  double winRate = 0.0;
  int outcomeCount = 0;

  // --- pipeline / window ----------------------------------------------------
  WindowInfo? window;
  bool nextWindowReady = false;
  Map<String, dynamic> brainExplain = const {};
  Map<String, dynamic> wiring = const {};

  // --- emergency -----------------------------------------------------------
  Map<String, dynamic>? emergency;
  double emergencyRemaining = 0;

  /// ONE clock.  The backend publishes the window as two absolute instants plus
  /// its own time; we measure the offset between its clock and ours and then
  /// count down to a fixed point in time.  Within a window the deadline never
  /// moves, so nothing that arrives on the socket can make the number jump,
  /// repeat or restart - and every panel refreshes on the same grid marks.
  MasterClock? clock;
  int? _serverOffsetMs;
  int _endsAtMs = 0;
  int _startedAtMs = 0;
  int _windowId = -1;

  /// Client milliseconds translated onto the server's clock.
  int serverNowMs() => DateTime.now().millisecondsSinceEpoch + (_serverOffsetMs ?? 0);

  /// The refresh marks of the current window, and how long until the next one.
  ClockTick? get nextTick => clock?.nextTick;
  String? _lastPrediction;

  int? _lastSecondShown;
  int? _lastEmergencySecond;

  Timer? _countdown;
  Timer? _safetyNet;
  StreamSubscription<SignalEvent>? _subscription;

  bool get isComputing => lockState == 'COMPUTING';
  bool get isEmergency => lockState == 'EMERGENCY_OVERRIDE';
  bool get isLocked => lockState == 'LOCKED';
  /// The Signal Lock Protocol state, phrased for the pipelined engine: while a
  /// window is locked the engine is already working on the next one.
  String get lockLabel => isEmergency
      ? 'EMERGENCY OVERRIDE'
      : isLocked
          ? (nextWindowReady
              ? 'LOCKED - next window computed and held until the boundary'
              : 'LOCKED - computing the next window now')
          : 'COMPUTING...';

  // =========================================================================
  // Lifecycle
  // =========================================================================
  Future<void> start() async {
    final prefs = await SharedPreferences.getInstance();
    asset = prefs.getString('asset') ?? 'BTC';

    final config = await api.config();
    if (config != null) {
      cyclePeriodSeconds = config.cyclePeriodSeconds;
      lockDeadlineSeconds = config.lockDeadlineSeconds;
      weights = config.weights;
      if (!config.assets.contains(asset) && config.assets.isNotEmpty) {
        asset = config.assets.first;
      }
    }

    catalog = await api.formulaCatalog();
    await refreshStatus();
    await refreshNews();
    await refreshAgents();
    await refreshBrain();
    await refreshWiring();
    await refreshBrainExplain();
    await refreshHistory();
    await refreshOutcomes();
    await refreshTimings();

    _subscription = socket.events.listen(_onEvent);
    socket.connect();

    // ONE frame-rate tick drives every clock readout; the panels are driven by
    // the messages the backend schedules on the window grid (SIGNAL, PULSE).
    _countdown =
        Timer.periodic(const Duration(milliseconds: 100), (_) => _tick());

    // ONE safety net, and it only acts while the socket is down.
    _safetyNet = Timer.periodic(const Duration(seconds: 5), (_) {
      if (!connected) refreshPanels();
    });
    notifyListeners();
  }

  @override
  void dispose() {
    _countdown?.cancel();
    _safetyNet?.cancel();
    _subscription?.cancel();
    socket.dispose();
    super.dispose();
  }

  void _tick() {
    // The countdown is derived from the absolute window end the server
    // published, so it is monotonic inside a window by construction.
    if (_endsAtMs > 0) {
      secondsRemaining =
          ((_endsAtMs - serverNowMs()) / 1000).clamp(0.0, cyclePeriodSeconds);
    }
    if (emergencyRemaining > 0) {
      emergencyRemaining = (emergencyRemaining - 0.1).clamp(0, 600);
      if (emergencyRemaining == 0) emergency = null;
    }
    // Repaint only when something the user can read actually changed: the whole
    // second, or the emergency countdown.  A 10 Hz rebuild of every widget is
    // what "glitchy" looks like on a phone.
    final second = secondsRemaining.ceil();
    final emergencySecond = emergencyRemaining.ceil();
    if (second != _lastSecondShown || emergencySecond != _lastEmergencySecond) {
      _lastSecondShown = second;
      _lastEmergencySecond = emergencySecond;
      notifyListeners();
    }
  }

  // =========================================================================
  // WebSocket
  // =========================================================================
  void _onEvent(SignalEvent event) {
    switch (event.type) {
      case 'SOCKET_OPEN':
        connected = true;
        break;
      case 'SOCKET_CLOSED':
        connected = false;
        break;
      case 'HELLO':
        asset = (event.data['asset'] ?? asset).toString();
        pendingAsset = event.data['pending_asset']?.toString();
        _applyWindow(event.data['window'] ?? _statusWindow(event.data['status']));
        _applySignal(event.data['signal']);
        _applyStatus(event.data['status']);
        refreshWiring();
        refreshBrainExplain();
        break;
      case 'CYCLE_START':
        // Only the non-pipelined mode sends this.  Even then the previous
        // signal is kept on screen: a blank panel is never acceptable.
        cycleNumber = (event.data['cycle_number'] as num?)?.toInt() ?? cycleNumber;
        asset = (event.data['asset'] ?? asset).toString();
        lockState = 'COMPUTING';
        lockIcon = '\u{23F3}';
        break;
      case 'SIGNAL':
        // The boundary message is a complete snapshot: adopting it repaints
        // every panel in this one pass, in parallel with the countdown flip.
        _applyWindow(event.data['window'] ?? event.data['clock']);
        _applySignal(event.data);
        _applySnapshot(event.data, includeHistory: true);
        break;
      case 'PULSE':
        // The heartbeat on the window grid: formulas, news, agents, brain and
        // accuracy all arrive together, so nothing refreshes on its own timer.
        _applyWindow(event.data['window']);
        _applyPulse(event.data);
        break;
      case 'NEXT_WINDOW_READY':
        // A pipeline flag only - the next tick of the countdown shows it.
        nextWindowReady = true;
        break;
      case 'FORMULA_UPDATE':
        final raw = (event.data['formulas'] as Map?) ?? const {};
        liveFormulas = raw.map((k, v) => MapEntry(k.toString(), _num(v)));
        break;
      case 'EMERGENCY_OVERRIDE':
        emergency = event.data;
        emergencyRemaining =
            (event.data['remaining_seconds'] as num?)?.toDouble() ?? 0;
        _applySignal(event.data['signal']);
        break;
      case 'OUTCOME':
        final outcome = SignalOutcome.fromJson(event.data);
        outcomes = [outcome, ...outcomes].take(12).toList();
        winRate = outcome.winRate;
        break;
      case 'ASSET_SWITCH':
        pendingAsset = event.data['pending']?.toString();
        break;
    }
    notifyListeners();
  }

  /// Haptics: the user should feel a new window arrive, not have to stare at
  /// the screen.  A direction change is a medium impact, an emergency override
  /// is a heavy double buzz, and the very first lock is a light selection tick.
  void _hapticFor(String? previous, FrozenSignal next) {
    if (previous == null) {
      HapticFeedback.selectionClick();
    } else if (previous == next.signal) {
      return;
    } else if (next.isEmergencyOverride) {
      HapticFeedback.heavyImpact();
      HapticFeedback.vibrate();
      Future<void>.delayed(const Duration(milliseconds: 140), () {
        HapticFeedback.mediumImpact();
      });
    } else {
      HapticFeedback.mediumImpact();
    }
  }

  /// Apply a pulse: every panel it carries, in one pass.
  void _applyPulse(Map<String, dynamic> data) {
    final live = data['live_formulas'];
    if (live is Map) {
      final raw = (live['formulas'] as Map?) ?? const {};
      liveFormulas = raw.map((k, v) => MapEntry(k.toString(), _num(v)));
      final timings = live['timings_ms'];
      if (timings is Map) {
        timingsMap = Map<String, dynamic>.from(timings);
        totalFormulaMs = _num(live['total_ms']);
      }
    }
    _applySnapshot(data, includeHistory: false);
  }

  /// The shared parts of a snapshot: news, agents, brain, accuracy, history.
  void _applySnapshot(Map<String, dynamic> data, {bool includeHistory = false}) {
    final newsFeed = data['news_feed'];
    if (newsFeed is Map) {
      news = NewsSnapshot.fromJson(Map<String, dynamic>.from(newsFeed));
    }
    final agents = data['agents_status'];
    if (agents is Map) {
      agentStatus = Map<String, dynamic>.from(agents);
      final rawWeights = agents['weights'];
      if (rawWeights is Map) {
        weights = rawWeights.map((k, v) => MapEntry(k.toString(), _num(v)));
      }
    }
    final brain = data['brain_explain'];
    if (brain is Map && brain['available'] != false) {
      brainExplain = Map<String, dynamic>.from(brain);
    }
    final accuracy = data['accuracy'];
    if (accuracy is Map) {
      winRate = _num(accuracy['win_rate'], winRate);
      outcomeCount = (accuracy['evaluated'] as num?)?.toInt() ?? outcomeCount;
      final perSide = accuracy['per_side'];
      if (perSide is Map) {
        buyAccuracy = Map<String, dynamic>.from(
            (perSide['BUY'] as Map?) ?? const {});
        sellAccuracy = Map<String, dynamic>.from(
            (perSide['SELL'] as Map?) ?? const {});
      }
    }
    if (includeHistory) {
      final rows = data['history'];
      if (rows is Map && rows['history'] is List) {
        history = (rows['history'] as List)
            .map((row) => Map<String, dynamic>.from(row as Map))
            .toList();
      } else if (rows is List) {
        history = rows
            .map((row) => Map<String, dynamic>.from(row as Map))
            .toList();
      }
      final outcomesPayload = data['outcomes'];
      if (outcomesPayload is Map && outcomesPayload['rows'] is List) {
        outcomes = (outcomesPayload['rows'] as List)
            .map((row) => SignalOutcome.fromJson(Map<String, dynamic>.from(row as Map)))
            .toList()
            .reversed
            .take(12)
            .toList();
        winRate = _num(outcomesPayload['win_rate'], winRate);
        outcomeCount =
            (outcomesPayload['count'] as num?)?.toInt() ?? outcomeCount;
      }
    }
  }

  void _applySignal(dynamic raw) {
    if (raw is! Map) return;
    if (raw['signal'] == null) {
      // Sentinel from a cold start: keep the layout, wait for the first lock.
      _applyWindow(raw['window']);
      return;
    }
    final parsed = FrozenSignal.fromJson(Map<String, dynamic>.from(raw));
    final previous = signal?.signal ?? _lastPrediction;
    if (parsed.window != null) _applyWindow(raw['window']);
    _hapticFor(previous, parsed);
    _lastPrediction = parsed.signal;
    signal = parsed;
    lockState = parsed.lockState;
    lockIcon = parsed.lockIcon;
    holdWarning = parsed.holdWarning;
    prediction = parsed.prediction;
    predictionReceivedAtMs = serverNowMs();
    lastLockedFormulas = parsed.formulas;
    if (parsed.formulas.isNotEmpty && liveFormulas.isEmpty) {
      liveFormulas = parsed.formulas;
    }
  }

  /// Adopt the window block and its clock.
  ///
  /// The countdown re-anchors **only when the window changes** (a new cycle id).
  /// Any other message that happens to carry a window block - a pulse, a status
  /// poll, a reconnect - just refreshes the descriptive fields, so a late
  /// message can never move the deadline under a running countdown.
  void _applyWindow(dynamic raw) {
    if (raw is! Map) return;
    final parsed = WindowInfo.fromJson(Map<String, dynamic>.from(raw));
    window = parsed;
    if (parsed.windowSeconds > 0) cyclePeriodSeconds = parsed.windowSeconds;
    nextWindowReady = parsed.prefetchReady;

    final incoming = parsed.clock ?? _clockFromWindow(parsed);
    if (incoming == null) {
      // Very old payload: fall back to the reading it carried.
      secondsRemaining = parsed.secondsRemaining.clamp(0.0, cyclePeriodSeconds);
      return;
    }
    clock = incoming;
    cyclePeriodSeconds = incoming.windowSeconds > 0
        ? incoming.windowSeconds
        : cyclePeriodSeconds;

    // Nudge the offset estimate towards the server's clock instead of snapping:
    // a slow network must not shift the deadline.
    final sample = incoming.serverTimeMs - DateTime.now().millisecondsSinceEpoch;
    final current = _serverOffsetMs;
    _serverOffsetMs = current == null
        ? sample
        : current + (sample - current).clamp(-120, 120);

    if (incoming.cycleId != _windowId) {
      _windowId = incoming.cycleId;
      _startedAtMs = incoming.windowStartedAtMs;
      _endsAtMs = incoming.windowEndsAtMs;
      _lastSecondShown = null;
    }
    secondsRemaining =
        ((_endsAtMs - serverNowMs()) / 1000).clamp(0.0, cyclePeriodSeconds);
  }

  /// Older payloads carry the window without a nested clock block.
  MasterClock? _clockFromWindow(WindowInfo parsed) {
    if (parsed.secondsRemaining <= 0) return null;
    final now = DateTime.now().millisecondsSinceEpoch;
    return MasterClock(
      windowStartedAtMs: now,
      windowEndsAtMs: now + (parsed.secondsRemaining * 1000).round(),
      serverTimeMs: now,
      cycleId: _windowId + 1,
      periodSeconds: parsed.windowSeconds,
      windowSeconds: parsed.windowSeconds,
      secondsRemaining: parsed.secondsRemaining,
    );
  }

  /// What the countdown is counting down to, for the sub-line under it.
  String get countdownNote {
    final tick = nextTick;
    final cadence = clock?.minuteAligned == true
        ? '60s window, minute-aligned'
        : '${cyclePeriodSeconds.round()}s window';
    if (tick == null) {
      return 'every panel refreshes together at the boundary · $cadence';
    }
    final parts = tick.parts.join(' + ');
    return 'next refresh t+${tick.offsetSeconds.round()}s in '
        '${tick.secondsUntil.ceil()}s · $parts · $cadence';
  }

  static dynamic _statusWindow(dynamic status) =>
      status is Map ? status['window'] : null;

  /// Progress of the *next* window's computation, 0..1.
  double get nextComputeProgress {
    if (nextWindowReady) return 1.0;
    final w = window;
    if (w == null || !w.pipeline) return 0.0;
    final lead = _lockDeadlineSeconds <= 0 ? 8.0 : _lockDeadlineSeconds;
    return ((lead - w.secondsRemaining) / lead).clamp(0.0, 1.0);
  }

  double _lockDeadlineSeconds = 8.0;

  /// How long before the boundary the engine starts computing the next signal.
  set lockDeadlineSeconds(double value) => _lockDeadlineSeconds = value;
  double get lockDeadlineSeconds => _lockDeadlineSeconds;

  /// Win/loss streak over the evaluated windows (positive = wins).
  int get outcomeStreak {
    var streak = 0;
    for (final row in outcomes.reversed) {
      if (row.outcome == 0) break;
      final sign = row.outcome > 0 ? 1 : -1;
      if (streak == 0) {
        streak = sign;
      } else if ((streak > 0 ? 1 : -1) == sign) {
        streak += sign;
      } else {
        break;
      }
    }
    return streak;
  }

  SignalOutcome? get lastOutcome => outcomes.isEmpty ? null : outcomes.first;

  void _applyStatus(dynamic raw) {
    if (raw is! Map) return;
    final status = Map<String, dynamic>.from(raw);
    cycleNumber = (status['cycle']?['cycle_number'] as num?)?.toInt() ?? cycleNumber;
    degradationLevel =
        (status['degradation_level'] as num?)?.toInt() ?? degradationLevel;
    degradationLabel = (status['degradation_label'] ?? '').toString();
    final worldClock = status['clock'];
    if (worldClock is Map) {
      ntpSynced = worldClock['ntp_synced'] != false;
    }
    // The window block carries the master clock; adopting it is what anchors
    // the countdown (and it only re-anchors when the window id changes).
    _applyWindow(status['window']);
    final lock = status['lock'];
    if (lock is Map) {
      lockState = (lock['state'] ?? lockState).toString();
      lockIcon = (lock['icon'] ?? lockIcon).toString();
      if (lock['emergency_active'] == true) {
        emergencyRemaining =
            (lock['emergency_remaining'] as num?)?.toDouble() ?? 0;
      }
    }
    final infra = status['infrastructure'];
    if (infra is Map) {
      winRate = (infra['win_rate'] as num?)?.toDouble() ?? winRate;
      outcomeCount = (infra['outcomes'] as num?)?.toInt() ?? outcomeCount;
    }
  }

  // =========================================================================
  // REST refreshes
  // =========================================================================
  Future<void> refreshStatus() async {
    final status = await api.signalStatus();
    if (status != null) _applyStatus(status);
    notifyListeners();
  }

  Future<void> refreshNews() async {
    final snapshot = await api.news();
    if (snapshot != null) news = snapshot;
    notifyListeners();
  }

  Future<void> refreshAgents() async {
    final json = await api.agents();
    if (json != null) {
      agentStatus = json;
      final rawWeights = json['weights'];
      if (rawWeights is Map) {
        weights = rawWeights.map((k, v) => MapEntry(k.toString(), _num(v)));
      }
    }
    notifyListeners();
  }

  Future<void> refreshBrain() async {
    final json = await api.brainStatus();
    if (json != null) brainStatus = json;
    notifyListeners();
  }

  /// The static map: which formula drives which neuron (item 6).
  Future<void> refreshWiring() async {
    final json = await api.brainWiring();
    if (json != null) wiring = json;
    notifyListeners();
  }

  /// The live per-window story of the circuit.
  Future<void> refreshBrainExplain() async {
    final json = await api.brainExplain();
    if (json != null) brainExplain = json;
    notifyListeners();
  }

  Future<void> refreshHistory() async {
    final json = await api.history(limit: 12);
    if (json != null) {
      history = ((json['history'] as List?) ?? const [])
          .map((row) => Map<String, dynamic>.from(row as Map))
          .toList();
    }
    notifyListeners();
  }

  Future<void> refreshOutcomes() async {
    final json = await api.outcomes();
    if (json != null) {
      final rows = ((json['rows'] as List?) ?? const [])
          .map((row) => SignalOutcome.fromJson(Map<String, dynamic>.from(row as Map)))
          .toList()
          .reversed
          .toList();
      outcomes = rows.take(12).toList();
      winRate = _num(json['win_rate']);
      outcomeCount = (json['count'] as num?)?.toInt() ?? rows.length;
    }
    notifyListeners();
  }

  Future<void> refreshTimings() async {
    final json = await api.timings();
    if (json != null) timings = json;
    notifyListeners();
  }

  Future<void> refreshLiveFormulas() async {
    final json = await api.formulasLive();
    final raw = json?['formulas'];
    if (raw is Map) {
      liveFormulas = raw.map((k, v) => MapEntry(k.toString(), _num(v)));
      notifyListeners();
    }
  }

  Future<void> refreshPanels() async {
    await refreshStatus();
    await refreshAgents();
    await refreshNews();
    await refreshOutcomes();
    await refreshBrainExplain();
    if (isEmergency) await refreshBrain();
  }

  // =========================================================================
  // Actions
  // =========================================================================
  Future<void> selectAsset(String next) async {
    if (next == asset) return;
    pendingAsset = next;
    notifyListeners();
    socket.switchAsset(next);
    await api.switchAsset(next);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('asset', next);
  }

  Future<void> clearEmergency() async {
    await api.clearEmergency();
    emergency = null;
    emergencyRemaining = 0;
    if (lockState == 'EMERGENCY_OVERRIDE') lockState = 'LOCKED';
    notifyListeners();
  }

  Future<void> pollNews() async {
    await api.pollNews();
    await refreshNews();
  }

  Future<void> reconnectBrain() async {
    await api.brainReconnect();
    await refreshBrain();
  }

  Future<void> loadStubModel() async {
    await api.localModelStub();
    await refreshAgents();
  }

  static double _num(dynamic value) {
    if (value is num) return value.toDouble();
    return double.tryParse('$value') ?? 0.0;
  }
}
