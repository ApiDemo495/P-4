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

/// The verbatim Section 10.2 HOLD box, plus the lean the engine kept.
class HoldWarning {
  const HoldWarning({
    required this.text,
    this.lean,
    this.leanScore = 0.0,
    this.confidence = 0.0,
  });

  final String text;
  final String? lean;
  final double leanScore;
  final double confidence;

  factory HoldWarning.fromJson(Map<String, dynamic> json) => HoldWarning(
        text: _s(json['text']),
        lean: json['lean'] == null ? null : _s(json['lean']),
        leanScore: _d(json['lean_score']),
        confidence: _d(json['confidence']),
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
  });

  final int cycleNumber;
  final String timestamp;
  final String asset;
  final String signal; // BUY | SELL | HOLD
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

  bool get isBuy => signal == 'BUY';
  bool get isSell => signal == 'SELL';
  bool get isHold => signal == 'HOLD';

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
      signal: _s(json['signal'], 'HOLD'),
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
      holdWarning: json['hold_warning'] == null
          ? null
          : HoldWarning.fromJson(
              Map<String, dynamic>.from(json['hold_warning'] as Map)),
      weightsUsed:
          rawWeights.map((k, v) => MapEntry(k.toString(), _d(v))),
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
