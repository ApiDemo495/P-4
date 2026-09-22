import 'dart:async';

import 'package:flutter/foundation.dart';
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
  double winRate = 0.0;
  int outcomeCount = 0;

  // --- emergency -----------------------------------------------------------
  Map<String, dynamic>? emergency;
  double emergencyRemaining = 0;

  Timer? _countdown;
  Timer? _poll;
  StreamSubscription<SignalEvent>? _subscription;

  bool get isComputing => lockState == 'COMPUTING';
  bool get isEmergency => lockState == 'EMERGENCY_OVERRIDE';
  bool get isLocked => lockState == 'LOCKED';
  String get lockLabel => isEmergency
      ? 'EMERGENCY OVERRIDE - HOLD'
      : isLocked
          ? 'LOCKED - immutable for the rest of the cycle'
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
    await refreshHistory();
    await refreshOutcomes();
    await refreshTimings();

    _subscription = socket.events.listen(_onEvent);
    socket.connect();

    _countdown =
        Timer.periodic(const Duration(milliseconds: 100), (_) => _tick());
    _poll = Timer.periodic(const Duration(seconds: 15), (_) => refreshPanels());
    notifyListeners();
  }

  @override
  void dispose() {
    _countdown?.cancel();
    _poll?.cancel();
    _subscription?.cancel();
    socket.dispose();
    super.dispose();
  }

  void _tick() {
    if (secondsRemaining > 0) {
      secondsRemaining = (secondsRemaining - 0.1).clamp(0, cyclePeriodSeconds);
    }
    if (emergencyRemaining > 0) {
      emergencyRemaining = (emergencyRemaining - 0.1).clamp(0, 600);
      if (emergencyRemaining == 0) emergency = null;
    }
    notifyListeners();
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
        _applySignal(event.data['signal']);
        _applyStatus(event.data['status']);
        break;
      case 'CYCLE_START':
        cycleNumber = (event.data['cycle_number'] as num?)?.toInt() ?? cycleNumber;
        asset = (event.data['asset'] ?? asset).toString();
        lockState = 'COMPUTING';
        lockIcon = '\u{23F3}';
        signal = null;
        holdWarning = null;
        secondsRemaining = cyclePeriodSeconds;
        break;
      case 'SIGNAL':
        _applySignal(event.data);
        refreshHistory();
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

  void _applySignal(dynamic raw) {
    if (raw is! Map) return;
    final parsed = FrozenSignal.fromJson(Map<String, dynamic>.from(raw));
    signal = parsed;
    lockState = parsed.lockState;
    lockIcon = parsed.lockIcon;
    holdWarning = parsed.holdWarning;
    lastLockedFormulas = parsed.formulas;
    if (parsed.formulas.isNotEmpty && liveFormulas.isEmpty) {
      liveFormulas = parsed.formulas;
    }
  }

  void _applyStatus(dynamic raw) {
    if (raw is! Map) return;
    final status = Map<String, dynamic>.from(raw);
    cycleNumber = (status['cycle']?['cycle_number'] as num?)?.toInt() ?? cycleNumber;
    degradationLevel =
        (status['degradation_level'] as num?)?.toInt() ?? degradationLevel;
    degradationLabel = (status['degradation_label'] ?? '').toString();
    final clock = status['clock'];
    if (clock is Map) {
      ntpSynced = clock['ntp_synced'] != false;
      cyclePeriodSeconds =
          (clock['cycle_period_seconds'] as num?)?.toDouble() ?? cyclePeriodSeconds;
      final intoMinute =
          (clock['seconds_into_minute'] as num?)?.toDouble() ?? 0.0;
      secondsRemaining = (cyclePeriodSeconds - intoMinute).clamp(0, cyclePeriodSeconds);
    }
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
