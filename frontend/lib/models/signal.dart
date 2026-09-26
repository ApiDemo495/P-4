/// Wire models for the DROSOPHILA TRADER v2.0 signal protocol.
///
/// These mirror `FrozenSignal.to_dict()` in ``backend/core/signal_lock.py`` and
/// the WebSocket messages emitted by ``backend/api/routes_signals.py``.
library;

double _d(dynamic value, [double fallback = 0.0]) {
  if (value == null) return fallback;
  if (value is num) return value.toDouble();
  return double.tryParse(value.toString()) ?? fallback;
}

int _i(dynamic value, [int fallback = 0]) {
  if (value == null) return fallback;
  if (value is num) return value.toInt();
  return int.tryParse(value.toString()) ?? fallback;
}

String _s(dynamic value, [String fallback = '']) =>
    value == null ? fallback : value.toString();

/// The conviction note (the Section 10.2 box, re-purposed).
///
/// It is no longer about a missing signal - every window is BUY or SELL - but
/// about *size*: it appears when the side came from the tie-break ladder or
/// when the engine's conviction is below HIGH.
class HoldWarning {
  const HoldWarning({
    required this.text,
    this.lean,
    this.leanScore = 0.0,
    this.confidence = 0.0,
    this.direction = '',
    this.source = '',
    this.conviction = '',
  });

  final String text;
  final String? lean;
  final double leanScore;
  final double confidence;

  /// The side the note is about (BUY / SELL) and the reason it fired.
  final String direction;
  final String source;
  final String conviction;

  factory HoldWarning.fromJson(Map<String, dynamic> json) => HoldWarning(
        text: _s(json['text']),
        lean: json['lean'] == null ? null : _s(json['lean']),
        leanScore: _d(json['lean_score'] ?? json['edge']),
        confidence: _d(json['confidence']),
        direction: _s(json['direction']),
        source: _s(json['source']),
        conviction: _s(json['conviction']),
      );
}

/// One row of the agent table.
class AgentDecision {
  const AgentDecision({
    required this.name,
    this.decision,
    this.confidence,
    this.status = 'DISABLED',
    this.model = '',
    this.reasoning = '',
    this.error = '',
    this.weight = 0.0,
  });

  final String name;
  final String? decision;
  final double? confidence;
  final String status;
  final String model;
  final String reasoning;
  final String error;
  final double weight;

  bool get isActive => status == 'ACTIVE' || status == 'STUB';

  factory AgentDecision.fromJson(String name, Map<String, dynamic> json) =>
      AgentDecision(
        name: name,
        decision: json['decision'] == null ? null : _s(json['decision']),
        confidence:
            json['confidence'] == null ? null : _d(json['confidence']),
        status: _s(json['status'], 'DISABLED'),
        model: _s(json['model']),
        reasoning: _s(json['reasoning']),
        error: _s(json['error']),
      );
}

/// Take-profit / stop-loss geometry attached to every locked signal
/// (Section 10.5). Levels are derived from realised volatility, not from a
/// fixed pip target, and the take-profit is the same distance from the entry
/// as the stop-loss (1:1), so the ratio is always 1.00.
class SignalRisk {
  const SignalRisk({
    this.tradeable = false,
    // No default side: the protocol is binary (BUY or SELL), so an absent
    // direction is unknown rather than a third state.
    this.direction = '',
    this.entry = 0.0,
    this.takeProfit,
    this.stopLoss,
    this.tpBps = 0.0,
    this.slBps = 0.0,
    this.rr = 0.0,
    this.rrTarget = 1.0,
    this.volatilityBps = 0.0,
    this.horizonSeconds = 60.0,
    this.note = '',
  });

  final bool tradeable;
  final String direction;
  final double entry;
  final double? takeProfit;
  final double? stopLoss;
  final double tpBps;
  final double slBps;
  final double rr;

  /// The reward:risk the engine is aiming for.  1.0 = take-profit and
  /// stop-loss are the same distance from the entry.
  final double rrTarget;
  final double volatilityBps;
  final double horizonSeconds;
  final String note;

  static const SignalRisk none = SignalRisk();

  factory SignalRisk.fromJson(Map<String, dynamic> json) => SignalRisk(
        tradeable: json['tradeable'] == true,
        direction: _s(json['direction']),
        entry: _d(json['entry']),
        takeProfit:
            json['take_profit'] == null ? null : _d(json['take_profit']),
        stopLoss: json['stop_loss'] == null ? null : _d(json['stop_loss']),
        tpBps: _d(json['tp_bps']),
        slBps: _d(json['sl_bps']),
        rr: _d(json['rr']),
        rrTarget: _d(json['rr_target'], 1.0),
        volatilityBps: _d(json['volatility_bps']),
        horizonSeconds: _d(json['horizon_seconds'], 60),
        note: _s(json['note']),
      );
}

/// One line of the case for (or against) the side.
class ReasoningBullet {
  const ReasoningBullet({
    this.kind = '',
    this.text = '',
    this.supports = true,
  });

  final String kind;
  final String text;
  final bool supports;

  factory ReasoningBullet.fromJson(Map<String, dynamic> json) => ReasoningBullet(
        kind: _s(json['kind']),
        text: _s(json['text']),
        supports: json['supports'] != false,
      );
}

/// Why the side was chosen: a one-line summary plus the evidence both ways.
class PredictionReasoning {
  const PredictionReasoning({
    this.summary = '',
    this.bullets = const [],
    this.supports = const [],
    this.against = const [],
  });

  final String summary;
  final List<ReasoningBullet> bullets;
  final List<String> supports;
  final List<String> against;

  static const PredictionReasoning none = PredictionReasoning();

  factory PredictionReasoning.fromJson(Map<String, dynamic> json) {
    final raw = (json['bullets'] as List?) ?? const [];
    return PredictionReasoning(
      summary: _s(json['summary']),
      bullets: raw
          .whereType<Map>()
          .map((row) => ReasoningBullet.fromJson(Map<String, dynamic>.from(row)))
          .toList(),
      supports: ((json['supports'] as List?) ?? const [])
          .map((value) => value.toString())
          .toList(),
      against: ((json['against'] as List?) ?? const [])
          .map((value) => value.toString())
          .toList(),
    );
  }
}

/// The prediction the panel renders: the side, its fresh-ness, the 1:1 levels
/// and the reasoning behind it.
///
/// `ageSeconds` is the age as of the moment the payload was received; the UI
/// adds the time it has been on screen.  A prediction older than
/// `maxAgeSeconds` is stale by contract (15 s by default), and the chip says so.
class Prediction {
  const Prediction({
    this.side = '',
    this.confidence = 0.0,
    this.conviction = 'LOW',
    this.entry,
    this.takeProfit,
    this.stopLoss,
    this.tpBps = 0.0,
    this.slBps = 0.0,
    this.rr = 0.0,
    this.rrTarget = 1.0,
    this.volatilityBps = 0.0,
    this.horizonSeconds = 15.0,
    this.ageSeconds,
    this.maxAgeSeconds = 15.0,
    this.state = 'LIVE',
    this.computedAt = '',
    this.expiresAt = '',
    this.reasoning = PredictionReasoning.none,
  });

  final String side;
  final double confidence;
  final String conviction;
  final double? entry;
  final double? takeProfit;
  final double? stopLoss;
  final double tpBps;
  final double slBps;
  final double rr;
  final double rrTarget;
  final double volatilityBps;
  final double horizonSeconds;
  final double? ageSeconds;
  final double maxAgeSeconds;
  final String state; // LIVE | STALE
  final String computedAt;
  final String expiresAt;
  final PredictionReasoning reasoning;

  bool get isStale => state != 'LIVE';

  static const Prediction none = Prediction();

  factory Prediction.fromJson(Map<String, dynamic> json) => Prediction(
        side: _s(json['side']),
        confidence: _d(json['confidence']),
        conviction: _s(json['conviction'], 'LOW'),
        entry: json['entry'] == null ? null : _d(json['entry']),
        takeProfit:
            json['take_profit'] == null ? null : _d(json['take_profit']),
        stopLoss: json['stop_loss'] == null ? null : _d(json['stop_loss']),
        tpBps: _d(json['tp_bps']),
        slBps: _d(json['sl_bps']),
        rr: _d(json['rr']),
        rrTarget: _d(json['rr_target'], 1.0),
        volatilityBps: _d(json['volatility_bps']),
        horizonSeconds: _d(json['horizon_seconds'], 15),
        ageSeconds: json['age_seconds'] == null ? null : _d(json['age_seconds']),
        maxAgeSeconds: _d(json['max_age_seconds'], 15),
        state: _s(json['state'], 'LIVE'),
        computedAt: _s(json['computed_at']),
        expiresAt: _s(json['expires_at']),
        reasoning: json['reasoning'] == null
            ? PredictionReasoning.none
            : PredictionReasoning.fromJson(
                Map<String, dynamic>.from(json['reasoning'] as Map)),
      );
}

/// Which window the locked signal governs, and what the engine is doing in it.
///
/// The pipeline means `computedAt` is *always* inside the previous countdown:
/// the countdown the user reads shows a signal that already existed when the
/// window started, while the engine computes the next one.
class WindowInfo {
  const WindowInfo({
    this.validFrom = '',
    this.validUntil = '',
    this.secondsRemaining = 60.0,
    this.windowSeconds = 60.0,
    this.computedAt = '',
    this.computedSecondsAgo,
    this.computeMs = 0.0,
    this.prefetchReady = false,
    this.pipeline = true,
    this.phase = '',
    this.computeProgress = 0.0,
  });

  final String validFrom;
  final String validUntil;
  final double secondsRemaining;
  final double windowSeconds;
  final String computedAt;
  final double? computedSecondsAgo;
  final double computeMs;
  final bool prefetchReady;
  final bool pipeline;
  final String phase;
  final double computeProgress;

  factory WindowInfo.fromJson(Map<String, dynamic> json) => WindowInfo(
        validFrom: _s(json['valid_from']),
        validUntil: _s(json['valid_until']),
        secondsRemaining: _d(json['seconds_remaining'], 60),
        windowSeconds: _d(json['window_seconds'], 60),
        computedAt: _s(json['computed_at']),
        computedSecondsAgo: json['computed_seconds_ago'] == null
            ? null
            : _d(json['computed_seconds_ago']),
        computeMs: _d(json['compute_ms']),
        prefetchReady: json['prefetch_ready'] == true,
        pipeline: json['pipeline'] != false,
        phase: _s(json['phase']),
        computeProgress: _d(json['compute_progress']),
      );
}

/// The locked signal for one cycle. Immutable by design - the backend cannot
/// send a different one until the next cycle begins.
class FrozenSignal {
  const FrozenSignal({
    required this.cycleNumber,
    required this.timestamp,
    required this.asset,
    required this.signal,
    required this.confidence,
    required this.reasoning,
    required this.lockState,
    required this.lockIcon,
    required this.isEmergencyOverride,
    this.holdLean,
    this.ccsValue = 0.0,
    this.ccsConfidence = 0.0,
    this.drg = 0.0,
    this.brainStatus = '',
    this.degradationLevel = 1,
    this.warnings = const [],
    this.price = 0.0,
    this.totalMs = 0.0,
    this.emergencyHeadline = '',
    this.supersededBy,
    this.formulas = const {},
    this.hedge = const {},
    this.news = const {},
    this.agents = const {},
    this.holdWarning,
    this.weightsUsed = const {},
    this.risk = SignalRisk.none,
    this.prediction = Prediction.none,
    this.window,
    this.preview = false,
    this.computedAt = '',
    this.validFrom = '',
    this.validUntil = '',
    this.windowSeconds = 60.0,
  });

  final int cycleNumber;
  final String timestamp;
  final String asset;
  final String signal; // BUY | SELL - the protocol is binary
  final double confidence;
  final String reasoning;
  final String lockState; // COMPUTING | LOCKED | EMERGENCY_OVERRIDE
  final String lockIcon; // hourglass | padlock | bolt
  final bool isEmergencyOverride;
  final String? holdLean;
  final double ccsValue;
  final double ccsConfidence;
  final double drg;
  final String brainStatus;
  final int degradationLevel;
  final List<String> warnings;
  final double price;
  final double totalMs;
  final String emergencyHeadline;
  final String? supersededBy;
  final Map<String, double> formulas;
  final Map<String, dynamic> hedge;
  final Map<String, dynamic> news;
  final Map<String, AgentDecision> agents;
  final HoldWarning? holdWarning;
  final Map<String, double> weightsUsed;

  /// Take-profit / stop-loss block, 1:1 by contract.
  final SignalRisk risk;

  /// The prediction block: side, freshness, levels, reasoning and accuracy.
  final Prediction prediction;

  /// The window this signal governs (pipeline metadata).
  final WindowInfo? window;
  final bool preview;
  final String computedAt;
  final String validFrom;
  final String validUntil;
  final double windowSeconds;

  bool get isBuy => signal == 'BUY';
  bool get isSell => signal == 'SELL';
  /// Always false: HOLD was removed from the protocol.  Kept as a
  /// deprecated shim so an older widget cannot resurrect the third state.
  @Deprecated('HOLD was removed: every window is BUY or SELL')
  bool get isHold => false;

  double hedgeValue(String key) => _d(hedge[key]);

  factory FrozenSignal.fromJson(Map<String, dynamic> json) {
    final rawFormulas = (json['formulas'] as Map?) ?? const {};
    final rawHedge = (json['hedge'] as Map?) ?? const {};
    final rawNews = (json['news'] as Map?) ?? const {};
    final rawAgents = (json['agents'] as Map?) ?? const {};
    final fusion = (json['fusion'] as Map?) ?? const {};
    final rawWeights = (fusion['weights_used'] as Map?) ?? const {};

    return FrozenSignal(
      cycleNumber: _i(json['cycle_number']),
      timestamp: _s(json['timestamp']),
      asset: _s(json['asset'], 'BTC'),
      signal: _s(json['signal'], 'SELL'),
      confidence: _d(json['confidence']),
      reasoning: _s(json['reasoning']),
      lockState: _s(json['lock_state'], 'LOCKED'),
      lockIcon: _s(json['lock_icon'], '\u{1F512}'),
      isEmergencyOverride: json['is_emergency_override'] == true,
      holdLean: json['hold_lean'] == null ? null : _s(json['hold_lean']),
      ccsValue: _d(json['ccs_value']),
      ccsConfidence: _d(json['ccs_confidence']),
      drg: _d(json['drg']),
      brainStatus: _s(json['brain_status']),
      degradationLevel: _i(json['degradation_level'], 1),
      warnings: ((json['warnings'] as List?) ?? const [])
          .map((w) => w.toString())
          .toList(),
      price: _d(json['price']),
      totalMs: _d(json['total_ms']),
      emergencyHeadline: _s(json['emergency_headline']),
      supersededBy:
          json['superseded_by'] == null ? null : _s(json['superseded_by']),
      formulas: rawFormulas.map((k, v) => MapEntry(k.toString(), _d(v))),
      hedge: rawHedge.map((k, v) => MapEntry(k.toString(), v)),
      news: rawNews.map((k, v) => MapEntry(k.toString(), v)),
      agents: rawAgents.map(
        (k, v) => MapEntry(
          k.toString(),
          AgentDecision.fromJson(
              k.toString(), Map<String, dynamic>.from(v as Map)),
        ),
      ),
      holdWarning: json['conviction_note'] == null
          ? null
          : HoldWarning.fromJson(
              Map<String, dynamic>.from(json['conviction_note'] as Map)),
      weightsUsed:
          rawWeights.map((k, v) => MapEntry(k.toString(), _d(v))),
      prediction: json['prediction'] == null
          ? Prediction.none
          : Prediction.fromJson(
              Map<String, dynamic>.from(json['prediction'] as Map)),
      risk: json['risk'] == null
          ? SignalRisk.none
          : SignalRisk.fromJson(Map<String, dynamic>.from(json['risk'] as Map)),
      window: json['window'] == null
          ? null
          : WindowInfo.fromJson(Map<String, dynamic>.from(json['window'] as Map)),
      preview: json['preview'] == true,
      computedAt: _s(json['computed_at']),
      validFrom: _s(json['valid_from']),
      validUntil: _s(json['valid_until']),
      windowSeconds: _d(json['window_seconds'], 60),
    );
  }
}

/// A scored outcome from a previous cycle (`OUTCOME` message).
class SignalOutcome {
  const SignalOutcome({
    required this.cycleNumber,
    required this.signal,
    required this.outcome,
    required this.pnlBps,
    required this.winRate,
  });

  final int cycleNumber;
  final String signal;
  final double outcome;
  final double pnlBps;
  final double winRate;

  String get label => outcome > 0 ? 'WIN' : outcome < 0 ? 'LOSS' : 'FLAT';

  factory SignalOutcome.fromJson(Map<String, dynamic> json) => SignalOutcome(
        cycleNumber: _i(json['cycle_number']),
        signal: _s(json['signal']),
        outcome: _d(json['outcome']),
        pnlBps: _d(json['pnl_bps']),
        winRate: _d(json['win_rate']),
      );
}

/// One formula from `/api/formulas`, used by the explorer.
class FormulaSpec {
  const FormulaSpec({
    required this.index,
    required this.name,
    required this.title,
    required this.category,
    required this.brainNode,
    required this.description,
    required this.directional,
    required this.budgetMs,
  });

  final int index;
  final String name;
  final String title;
  final String category;
  final String brainNode;
  final String description;
  final bool directional;
  final double budgetMs;

  factory FormulaSpec.fromJson(Map<String, dynamic> json) => FormulaSpec(
        index: _i(json['index']),
        name: _s(json['name']),
        title: _s(json['title']),
        category: _s(json['category']),
        brainNode: _s(json['brain_node']),
        description: _s(json['description']),
        directional: json['directional'] != false,
        budgetMs: _d(json['latency_ms']),
      );
}

class FormulaCategory {
  const FormulaCategory({
    required this.key,
    required this.name,
    required this.formulas,
  });

  final String key;
  final String name;
  final List<FormulaSpec> formulas;

  factory FormulaCategory.fromJson(Map<String, dynamic> json) =>
      FormulaCategory(
        key: _s(json['key']),
        name: _s(json['name']),
        formulas: ((json['formulas'] as List?) ?? const [])
            .map((f) =>
                FormulaSpec.fromJson(Map<String, dynamic>.from(f as Map)))
            .toList(),
      );
}

/// News status block (`/api/news`).
class NewsSnapshot {
  const NewsSnapshot({
    this.headline = 'No headlines available',
    this.source = '',
    this.tier = 0,
    this.niv = 0.0,
    this.smd = 0.0,
    this.coverage = 'limited',
    this.ageSeconds,
    this.items = const [],
  });

  final String headline;
  final String source;
  final int tier;
  final double niv;
  final double smd;
  final String coverage;
  final double? ageSeconds;
  final List<NewsItem> items;

  factory NewsSnapshot.fromJson(Map<String, dynamic> json) {
    final status = (json['status'] as Map?) ?? const {};
    final items = ((json['items'] as List?) ?? const [])
        .map((i) => NewsItem.fromJson(Map<String, dynamic>.from(i as Map)))
        .toList();
    final first = items.isNotEmpty ? items.first : null;
    return NewsSnapshot(
      headline: first?.headline ?? 'No headlines available',
      source: first?.source ?? '',
      tier: first?.tier ?? 0,
      niv: _d(json['niv']),
      smd: _d(json['smd']),
      coverage: _s(status['coverage'], 'limited'),
      ageSeconds:
          json['seconds_since_poll'] == null ? null : _d(json['seconds_since_poll']),
      items: items,
    );
  }
}

class NewsItem {
  const NewsItem({
    required this.headline,
    required this.source,
    required this.tier,
    required this.sentiment,
    required this.ageSeconds,
  });

  final String headline;
  final String source;
  final int tier;
  final double sentiment;
  final double ageSeconds;

  factory NewsItem.fromJson(Map<String, dynamic> json) => NewsItem(
        headline: _s(json['headline']),
        source: _s(json['source']),
        tier: _i(json['tier']),
        sentiment: _d(json['sentiment']),
        ageSeconds: _d(json['age_seconds']),
      );
}

/// Subset of `/api/system/config` the client needs.
class SystemConfig {
  const SystemConfig({
    required this.assets,
    required this.cyclePeriodSeconds,
    required this.signalThreshold,
    required this.minConfidence,
    required this.lockDeadlineSeconds,
    required this.weights,
  });

  final List<String> assets;
  final double cyclePeriodSeconds;
  final double signalThreshold;
  final double minConfidence;
  final double lockDeadlineSeconds;
  final Map<String, double> weights;

  factory SystemConfig.fromJson(Map<String, dynamic> json) {
    final rawWeights = (json['weights'] as Map?) ?? const {};
    return SystemConfig(
      assets: ((json['assets'] as List?) ?? const ['BTC'])
          .map((a) => a.toString())
          .toList(),
      cyclePeriodSeconds: _d(json['cycle_period_seconds'], 60),
      signalThreshold: _d(json['signal_threshold'], 0.25),
      minConfidence: _d(json['min_fusion_confidence'], 0.55),
      lockDeadlineSeconds: _d(json['lock_deadline_seconds'], 8),
      weights: rawWeights.map((k, v) => MapEntry(k.toString(), _d(v))),
    );
  }
}
