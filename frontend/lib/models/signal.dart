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
/// `maxAgeSeconds` is stale by contract - its own 60 s window plus the 5 s
/// publishing grace - and the chip says so.
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
    this.horizonSeconds = 60.0,
    this.ageSeconds,
    this.maxAgeSeconds = 65.0,
    this.state = 'LIVE',
    this.computedAt = '',
    this.expiresAt = '',
    this.reasoning = PredictionReasoning.none,
    this.horizon = PredictionHorizon.none,
    this.detail = PredictionDetail.none,
    this.micro = MicroReading.none,
    this.forecastFor = '',
    this.ageMicroseconds = 0,
    this.ageLabel = '',
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

  /// The forecast window this prediction covers (Round I).
  final PredictionHorizon horizon;
  final PredictionDetail detail;
  final MicroReading micro;
  final String forecastFor;
  final int ageMicroseconds;
  final String ageLabel;

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
        horizonSeconds: _d(json['horizon_seconds'], 60),
        ageSeconds: json['age_seconds'] == null ? null : _d(json['age_seconds']),
        maxAgeSeconds: _d(json['max_age_seconds'], 65),
        state: _s(json['state'], 'LIVE'),
        computedAt: _s(json['computed_at']),
        expiresAt: _s(json['expires_at']),
        reasoning: json['reasoning'] == null
            ? PredictionReasoning.none
            : PredictionReasoning.fromJson(
                Map<String, dynamic>.from(json['reasoning'] as Map)),
        horizon: json['horizon'] == null
            ? PredictionHorizon.none
            : PredictionHorizon.fromJson(
                Map<String, dynamic>.from(json['horizon'] as Map)),
        detail: json['detail'] == null
            ? PredictionDetail.none
            : PredictionDetail.fromJson(
                Map<String, dynamic>.from(json['detail'] as Map)),
        micro: json['detail'] is Map && (json['detail'] as Map)['micro'] is Map
            ? MicroReading.fromJson(Map<String, dynamic>.from(
                (json['detail'] as Map)['micro'] as Map))
            : MicroReading.none,
        forecastFor: _s(json['forecast_for']),
        ageMicroseconds: (json['age_microseconds'] as num?)?.toInt() ?? 0,
        ageLabel: _s(json['age_label']),
      );
}

/// What the prediction is *for*: the forecast window it covers.
///
/// Round I makes this explicit.  A prediction is released at one instant and
/// covers the 60 seconds that follow it - not "now", and not "the window that
/// happens to be on screen".  The instants are epoch microseconds, so the panel
/// can print the release to the microsecond and count down to the target.
class PredictionHorizon {
  const PredictionHorizon({
    this.seconds = 60.0,
    this.label = 'the next 60 seconds',
    this.releasedAtPrecise = '',
    this.targetAtPrecise = '',
    this.releasedAtUs = 0,
    this.targetAtUs = 0,
    this.secondsToTarget = 0.0,
    this.microsecondsToTarget = 0,
    this.scoringSeconds = 60.0,
    this.scoredInSeconds = 0.0,
    this.scoredAtUs = 0,
    this.covers = '',
  });

  final double seconds;
  final String label;
  final String releasedAtPrecise;

  /// `HH:MM:SS.ffffff` - the release instant at the resolution the engine runs at.
  final String targetAtPrecise;
  final int releasedAtUs;
  final int targetAtUs;
  final double secondsToTarget;
  final int microsecondsToTarget;
  final double scoringSeconds;
  final double scoredInSeconds;

  /// When the outcome is measured, in epoch microseconds, so the panel can
  /// count down to it on the shared clock instead of printing a frozen number.
  final int scoredAtUs;
  final String covers;

  static const PredictionHorizon none = PredictionHorizon();

  /// `00:00:00.000000` -> `00:00:00.000` for the narrow panel line.
  String get releaseClock => releasedAtPrecise.isEmpty
      ? '—'
      : releasedAtPrecise.substring(releasedAtPrecise.indexOf('T') + 1,
          releasedAtPrecise.length).replaceAll('Z', '');

  factory PredictionHorizon.fromJson(Map<String, dynamic> json) =>
      PredictionHorizon(
        seconds: _d(json['seconds'], 60),
        label: _s(json['label'], 'the next 60 seconds'),
        releasedAtPrecise: _s(json['released_at_precise']),
        targetAtPrecise: _s(json['target_at_precise']),
        releasedAtUs: (json['released_at_us'] as num?)?.toInt() ?? 0,
        targetAtUs: (json['target_at_us'] as num?)?.toInt() ?? 0,
        secondsToTarget: _d(json['seconds_to_target']),
        microsecondsToTarget:
            (json['microseconds_to_target'] as num?)?.toInt() ?? 0,
        scoringSeconds: _d(json['scoring_seconds'], 60),
        scoredInSeconds: _d(json['scored_in_seconds']),
        scoredAtUs: (json['scored_at_us'] as num?)?.toInt() ?? 0,
        covers: _s(json['covers']),
      );
}

/// The tape measured in microseconds: how fast quotes arrive, how jittery that
/// arrival is, how long a quote survives and which side is being aggressive.
class MicroReading {
  const MicroReading({
    this.available = false,
    this.reason = '',
    this.ticks = 0,
    this.resolutionUs = 0,
    this.resolutionLabel = '',
    this.meanIntervalUs = 0,
    this.jitterUs = 0,
    this.tickRateHz = 0.0,
    this.quoteLifetimeUs = 0,
    this.aggression = 0.0,
    this.microVolBps = 0.0,
    this.lastTickAgeUs = 0,
  });

  final bool available;
  final String reason;
  final int ticks;
  final int resolutionUs;
  final String resolutionLabel;
  final int meanIntervalUs;
  final int jitterUs;
  final double tickRateHz;
  final int quoteLifetimeUs;
  final double aggression;
  final double microVolBps;
  final int lastTickAgeUs;

  bool get hasData => available && resolutionUs > 0;

  static const MicroReading none = MicroReading();

  factory MicroReading.fromJson(Map<String, dynamic> json) => MicroReading(
        available: json['available'] == true,
        reason: _s(json['reason']),
        ticks: (json['ticks'] as num?)?.toInt() ?? 0,
        resolutionUs: (json['resolution_us'] as num?)?.toInt() ?? 0,
        resolutionLabel: _s(json['resolution_label']),
        meanIntervalUs: (json['mean_interval_us'] as num?)?.toInt() ?? 0,
        jitterUs: (json['jitter_us'] as num?)?.toInt() ?? 0,
        tickRateHz: _d(json['tick_rate_hz']),
        quoteLifetimeUs: (json['quote_lifetime_us'] as num?)?.toInt() ?? 0,
        aggression: _d(json['aggression']),
        microVolBps: _d(json['micro_vol_bps']),
        lastTickAgeUs: (json['last_tick_age_us'] as num?)?.toInt() ?? 0,
      );
}

/// One named number inside the detail block (a supporting formula, a category
/// score, a confidence component).
class PredictionDetailRow {
  const PredictionDetailRow({this.name = '', this.value = 0.0, this.kind = ''});

  final String name;
  final double value;
  final String kind;

  factory PredictionDetailRow.fromJson(Map<String, dynamic> json) =>
      PredictionDetailRow(
        name: _s(json['name'], _s(json['key'])),
        value: _d(json['value']),
        kind: _s(json['kind']),
      );
}

/// The numbers behind the prediction, so the panel can explain it rather than
/// assert it: which formulas supported the side, which argued against, what
/// each category scored, how long the pass took in microseconds.
class PredictionDetail {
  const PredictionDetail({
    this.side = '',
    this.formulasEvaluated = 0,
    this.categoryScores = const {},
    this.supporters = const [],
    this.opponents = const [],
    this.confidenceParts = const {},
    this.computeUs = 0,
    this.publishLatencyUs = 0,
    this.tickIntervalUs = 0,
    this.resolutionUs = 0,
    this.historySamples = 0,
    this.micro = MicroReading.none,
    this.crowdDominant = '',
    this.crowdPercent = 0.0,
    this.crowdTimescale = '',
    this.crowdToneBias = 0.0,
    this.crowdManipulation = 0.0,
    this.crowdManipulationKind = '',
    this.crowdRead = '',
    this.crowdDampening = 1.0,
    this.crowdDeep = const {},
    this.crowdChain = const [],
    this.crowdAgreement = FormulaAgreement.none,
  });

  final String side;
  final int formulasEvaluated;
  final Map<String, double> categoryScores;
  final List<PredictionDetailRow> supporters;
  final List<PredictionDetailRow> opponents;
  final Map<String, double> confidenceParts;
  final int computeUs;
  final int publishLatencyUs;
  final int tickIntervalUs;
  final int resolutionUs;
  final int historySamples;
  final MicroReading micro;

  /// What the crowd was feeling when this side was locked (Round J).
  final String crowdDominant;
  final double crowdPercent;
  final String crowdTimescale;
  final double crowdToneBias;
  final double crowdManipulation;
  final String crowdManipulationKind;
  final String crowdRead;

  /// The confidence multiplier the crowd produced (1.0 = no dampening).
  final double crowdDampening;

  /// Round K: the deep-reasoning summary at lock time (belief, regime, VPIN,
  /// Hawkes n, Kyle's lambda, variance ratio, Hurst, entropy …) and the
  /// ordered reasoning chain.
  final Map<String, dynamic> crowdDeep;
  final List<DeepStep> crowdChain;

  /// Round L: the locked crowd against the lock-time formulas.
  final FormulaAgreement crowdAgreement;

  bool get hasData => side.isNotEmpty || supporters.isNotEmpty;
  bool get hasCrowd => crowdDominant.isNotEmpty;
  bool get hasDeep => crowdDeep['available'] == true;

  static const PredictionDetail none = PredictionDetail();

  static List<PredictionDetailRow> _rows(dynamic raw) =>
      ((raw as List?) ?? const [])
          .whereType<Map>()
          .map((row) => PredictionDetailRow.fromJson(Map<String, dynamic>.from(row)))
          .toList();

  static Map<String, double> _scores(dynamic raw) =>
      ((raw as Map?) ?? const {})
          .map((key, value) => MapEntry(key.toString(), _d(value)));

  factory PredictionDetail.fromJson(Map<String, dynamic> json) {
    final engine = (json['agreement'] as Map?)?['engine'] as Map?;
    final micro = json['micro'];
    final crowd = json['crowd'] as Map?;
    final parts = (json['confidence_parts'] as Map?) ?? const {};
    return PredictionDetail(
      side: _s(json['side']),
      formulasEvaluated:
          (json['formulas_evaluated'] as num?)?.toInt() ?? 0,
      categoryScores: _scores(json['category_scores']),
      supporters: _rows(json['supporters']),
      opponents: _rows(json['opponents']),
      confidenceParts: _scores(json['confidence_parts']),
      computeUs: (engine?['compute_us'] as num?)?.toInt() ?? 0,
      publishLatencyUs: (engine?['publish_latency_us'] as num?)?.toInt() ?? 0,
      tickIntervalUs: (engine?['tick_interval_us'] as num?)?.toInt() ?? 0,
      resolutionUs: (engine?['resolution_us'] as num?)?.toInt() ?? 0,
      historySamples: (engine?['history_samples'] as num?)?.toInt() ?? 0,
      micro: micro is Map
          ? MicroReading.fromJson(Map<String, dynamic>.from(micro))
          : MicroReading.none,
      crowdDominant: _s(crowd?['dominant']),
      crowdPercent: _d(crowd?['percent']),
      crowdTimescale: _s(crowd?['timescale']),
      crowdToneBias: _d(crowd?['tone_bias']),
      crowdManipulation: _d(crowd?['manipulation']),
      crowdManipulationKind: _s(crowd?['manipulation_kind']),
      crowdRead: _s(crowd?['read']),
      crowdDampening: _d(parts['crowd_dampening'], 1.0),
      crowdAgreement: crowd?['formula_agreement'] is Map
          ? FormulaAgreement.fromJson(
              Map<String, dynamic>.from(crowd!['formula_agreement'] as Map))
          : FormulaAgreement.none,
      crowdDeep: crowd?['deep'] is Map
          ? Map<String, dynamic>.from(crowd!['deep'] as Map)
          : const {},
      crowdChain: (((crowd?['deep'] as Map?)?['chain'] as List?) ?? const [])
          .whereType<Map>()
          .map((row) => DeepStep.fromJson(Map<String, dynamic>.from(row)))
          .toList(),
    );
  }
}

/// One of the crowd's eight emotions, with its intensity on each timescale.
///
/// Mirrors ``EmotionScore.to_dict()`` in ``backend/core/emotions.py``.
class EmotionScore {
  const EmotionScore({
    this.name = '',
    this.label = '',
    this.tone = 'neutral',
    this.family = '',
    this.intensity = 0.0,
    this.byTimescale = const {},
    this.dominantTimescale = '',
    this.drivers = const [],
    this.formula = '',
    this.terms = const [],
    this.gate = 1.0,
    this.ramp = 0.0,
    this.belief = 0.0,
  });

  final String name;
  final String label;

  /// Round L: the emotion's printed formula, its live terms
  /// (weight / term / value / contribution), the multiplicative gate, the raw
  /// ramp reading and the Bayesian filter's belief - so the number on the bar
  /// can be checked by hand.
  final String formula;
  final List<EmotionTerm> terms;
  final double gate;
  final double ramp;
  final double belief;

  /// `negative` (fear family), `positive` (chase family) or `neutral` (calm).
  final String tone;
  final String family;

  /// 0 … 1.
  final double intensity;

  /// micro / seconds / window / minutes / news → 0 … 1.
  final Map<String, double> byTimescale;
  final String dominantTimescale;
  final List<String> drivers;

  double get percent => intensity * 100.0;

  factory EmotionScore.fromJson(Map<String, dynamic> json) => EmotionScore(
        name: _s(json['name']),
        label: _s(json['label'], _s(json['name'])),
        tone: _s(json['tone'], 'neutral'),
        family: _s(json['family']),
        intensity: _d(json['intensity'], _d(json['percent']) / 100.0),
        byTimescale: ((json['by_timescale'] as Map?) ?? const {})
            .map((key, value) => MapEntry(key.toString(), _d(value))),
        dominantTimescale: _s(json['dominant_timescale']),
        drivers: ((json['drivers'] as List?) ?? const [])
            .map((item) => item.toString())
            .toList(),
        formula: _s(json['formula']),
        terms: ((json['terms'] as List?) ?? const [])
            .whereType<Map>()
            .map((row) => EmotionTerm.fromJson(Map<String, dynamic>.from(row)))
            .toList(),
        gate: _d(json['gate'], 1.0),
        ramp: _d(json['ramp']),
        belief: _d(json['belief']),
      );
}

/// One term of an emotion's formula: `weight · term = contribution`.
class EmotionTerm {
  const EmotionTerm({
    this.weight = 0.0,
    this.term = '',
    this.value = 0.0,
    this.contribution = 0.0,
  });

  final double weight;
  final String term;
  final double value;
  final double contribution;

  factory EmotionTerm.fromJson(Map<String, dynamic> json) => EmotionTerm(
        weight: _d(json['weight']),
        term: _s(json['term']),
        value: _d(json['value']),
        contribution: _d(json['contribution']),
      );
}

/// Round L: the crowd reading against the 22 formulas' weighted vote - the
/// verdict (aligned / conflict / crowd flat / formulas split / formulas
/// silent), both readings, and the rule that the formulas keep the vote.
class FormulaAgreement {
  const FormulaAgreement({
    this.available = false,
    this.consensus = 0.0,
    this.voters = 0,
    this.up = 0,
    this.down = 0,
    this.upNames = const [],
    this.downNames = const [],
    this.formulaSide = '',
    this.crowdTone = 0.0,
    this.crowdSide = '',
    this.alignment = 0.0,
    this.verdict = '',
    this.note = '',
    this.rule = '',
    this.formulaWeight = 0.40,
    this.crowdMaxCut = 0.25,
  });

  final bool available;
  final double consensus;
  final int voters;
  final int up;
  final int down;
  final List<String> upNames;
  final List<String> downNames;
  final String formulaSide;
  final double crowdTone;
  final String crowdSide;
  final double alignment;
  final String verdict;
  final String note;
  final String rule;
  final double formulaWeight;
  final double crowdMaxCut;

  static const FormulaAgreement none = FormulaAgreement();

  bool get hasVerdict => verdict.isNotEmpty;

  factory FormulaAgreement.fromJson(Map<String, dynamic> json) {
    final weights = (json['weights'] as Map?) ?? const {};
    List<String> names(dynamic raw) =>
        ((raw as List?) ?? const []).map((item) => item.toString()).toList();
    return FormulaAgreement(
      available: json['available'] == true,
      consensus: _d(json['consensus']),
      voters: _i(json['voters']),
      up: _i(json['up']),
      down: _i(json['down']),
      upNames: names(json['up_names']),
      downNames: names(json['down_names']),
      formulaSide: _s(json['formula_side']),
      crowdTone: _d(json['crowd_tone']),
      crowdSide: _s(json['crowd_side']),
      alignment: _d(json['alignment']),
      verdict: _s(json['verdict']),
      note: _s(json['note']),
      rule: _s(json['rule']),
      formulaWeight: _d(weights['formulas'], 0.40),
      crowdMaxCut: _d(weights['crowd_max_confidence_cut'], 0.125),
    );
  }
}

/// The manipulation signature of the minute: how crowded the tape looks and by
/// which mechanism (retail chase, stop hunt, whipsaw, book imbalance).
class ManipulationRead {
  const ManipulationRead({
    this.score = 0.0,
    this.kind = 'none',
    this.note = '',
    this.components = const {},
    this.evidence = const [],
  });

  final double score;
  final String kind;
  final String note;
  final Map<String, double> components;
  final List<String> evidence;

  static const ManipulationRead none = ManipulationRead();

  factory ManipulationRead.fromJson(Map<String, dynamic> json) =>
      ManipulationRead(
        score: _d(json['score']),
        kind: _s(json['kind'], 'none'),
        note: _s(json['note']),
        components: ((json['components'] as Map?) ?? const {})
            .map((key, value) => MapEntry(key.toString(), _d(value))),
        evidence: ((json['evidence'] as List?) ?? const [])
            .map((item) => item.toString())
            .toList(),
      );
}

/// One reading of the crowd: the eight emotions ranked, the dominant one, how
/// long it has held, the signed temperature and the manipulation read.
///
/// Mirrors ``EmotionReport.to_dict()`` (and its streamed ``compact`` form).
class EmotionReading {
  const EmotionReading({
    this.available = false,
    this.reason = '',
    this.asset = '',
    this.atUs = 0,
    this.ticks = 0,
    this.resolutionUs = 0.0,
    this.resolutionLabel = '',
    this.emotions = const [],
    this.dominant,
    this.runnerUp,
    this.toneBias = 0.0,
    this.heldSeconds = 0.0,
    this.churnPerMinute = 0,
    this.samples = 0,
    this.read = '',
    this.hint = '',
    this.intervalSeconds = 0.5,
    this.manipulation = ManipulationRead.none,
    this.deep = DeepReasoning.none,
    this.formulaAgreement = FormulaAgreement.none,
  });

  final bool available;
  final String reason;
  final String asset;
  final int atUs;
  final int ticks;
  final double resolutionUs;
  final String resolutionLabel;

  /// Ranked: the dominant emotion first.
  final List<EmotionScore> emotions;
  final EmotionScore? dominant;
  final EmotionScore? runnerUp;

  /// −1 terrified … +1 euphoric, 0 asleep.
  final double toneBias;
  final double heldSeconds;
  final int churnPerMinute;
  final int samples;
  final String read;
  final String hint;
  final double intervalSeconds;
  final ManipulationRead manipulation;

  /// Round K: the microstructure formulas and the Bayesian filter behind
  /// the reading.
  final DeepReasoning deep;

  /// Round L: how this reading sits against the 22 formulas.
  final FormulaAgreement formulaAgreement;

  static const EmotionReading none = EmotionReading();

  bool get hasData => available && emotions.isNotEmpty;

  /// The stream carries the heavy deep block on every fourth sample only;
  /// the samples in between keep the last one.
  EmotionReading withDeep(DeepReasoning carried) => EmotionReading(
        available: available,
        reason: reason,
        asset: asset,
        atUs: atUs,
        ticks: ticks,
        resolutionUs: resolutionUs,
        resolutionLabel: resolutionLabel,
        emotions: emotions,
        dominant: dominant,
        runnerUp: runnerUp,
        toneBias: toneBias,
        heldSeconds: heldSeconds,
        churnPerMinute: churnPerMinute,
        samples: samples,
        read: read,
        hint: hint,
        intervalSeconds: intervalSeconds,
        manipulation: manipulation,
        deep: carried,
        formulaAgreement: formulaAgreement,
      );

  factory EmotionReading.fromJson(Map<String, dynamic> json) {
    final ranked = ((json['emotions'] as List?) ?? const [])
        .whereType<Map>()
        .map((row) => EmotionScore.fromJson(Map<String, dynamic>.from(row)))
        .toList();
    EmotionScore? pick(dynamic raw) {
      if (raw is! Map) return null;
      final name = _s(raw['name']);
      for (final item in ranked) {
        if (item.name == name) return item;
      }
      return EmotionScore.fromJson(Map<String, dynamic>.from(raw));
    }

    final manipulation = json['manipulation'];
    return EmotionReading(
      available: json['available'] == true,
      reason: _s(json['reason']),
      asset: _s(json['asset']),
      atUs: _i(json['at_us']),
      ticks: _i(json['ticks']),
      resolutionUs: _d(json['resolution_us']),
      resolutionLabel: _s(json['resolution_label']),
      emotions: ranked,
      dominant: pick(json['dominant']) ?? (ranked.isNotEmpty ? ranked.first : null),
      runnerUp: pick(json['runner_up']) ?? (ranked.length > 1 ? ranked[1] : null),
      toneBias: _d(json['tone_bias']),
      heldSeconds: _d(json['held_seconds'], _d((json['dominant'] as Map?)?['held_seconds'])),
      churnPerMinute: _i(json['churn_per_minute']),
      samples: _i(json['samples']),
      read: _s(json['read']),
      hint: _s(json['hint']),
      intervalSeconds: _d(json['interval_seconds'], 0.5),
      manipulation: manipulation is Map
          ? ManipulationRead.fromJson(Map<String, dynamic>.from(manipulation))
          : ManipulationRead.none,
      deep: json['deep'] is Map
          ? DeepReasoning.fromJson(Map<String, dynamic>.from(json['deep'] as Map))
          : DeepReasoning.none,
      formulaAgreement: json['formula_agreement'] is Map
          ? FormulaAgreement.fromJson(
              Map<String, dynamic>.from(json['formula_agreement'] as Map))
          : FormulaAgreement.none,
    );
  }
}

/// One step of the deep reasoning chain (Round K): a formula, its value and
/// what it reads as, plus the emotions it argues for.
class DeepStep {
  const DeepStep({
    this.step = 0,
    this.name = '',
    this.formula = '',
    this.value = '',
    this.unit = '',
    this.reads = '',
    this.timescale = '',
    this.feeds = const [],
  });

  final int step;
  final String name;
  final String formula;
  final String value;
  final String unit;
  final String reads;
  final String timescale;
  final List<String> feeds;

  factory DeepStep.fromJson(Map<String, dynamic> json) {
    final raw = json['value'];
    return DeepStep(
      step: _i(json['step']),
      name: _s(json['name']),
      formula: _s(json['formula']),
      value: raw is num ? _trim(raw) : _s(raw),
      unit: _s(json['unit']),
      reads: _s(json['reads']),
      timescale: _s(json['timescale']),
      feeds: ((json['feeds'] as List?) ?? const []).map(_s).toList(),
    );
  }

  static String _trim(num v) {
    final d = v.toDouble();
    if (d == d.roundToDouble() && d.abs() < 1e6) return d.toStringAsFixed(0);
    return d.abs() < 0.01 ? d.toStringAsFixed(5) : d.toStringAsFixed(3);
  }
}

/// One of the three multi-scale bands (micro / seconds / window).
class DeepBand {
  const DeepBand({
    this.clock = '',
    this.hurst = 0.5,
    this.varianceRatio = 1.0,
    this.entropy = 1.0,
    this.samples = 0,
  });

  final String clock;
  final double hurst;
  final double varianceRatio;
  final double entropy;
  final int samples;

  factory DeepBand.fromJson(Map<String, dynamic> json) => DeepBand(
        clock: _s(json['clock']),
        hurst: _d(json['hurst'], 0.5),
        varianceRatio: _d(json['variance_ratio'], 1.0),
        entropy: _d(json['entropy'], 1.0),
        samples: _i(json['samples']),
      );
}

/// The deep microstructure layer (Round K): Hawkes self-excitation, VPIN,
/// Kyle's lambda, variance ratio / Hurst / entropy per band, sign memory,
/// wavelet spectrum, the regime filter, the manipulation detectors, the
/// Bayesian filter's posterior and the ordered reasoning chain.
class DeepReasoning {
  const DeepReasoning({
    this.available = false,
    this.reason = '',
    this.ticks = 0,
    this.computeUs = 0,
    this.branchingRatio = 0.0,
    this.intensityHz = 0.0,
    this.baselineHz = 0.0,
    this.vpin = 0.0,
    this.kyleLambdaBps = 0.0,
    this.kyleR2 = 0.0,
    this.impactNorm = 0.0,
    this.signMemory = 0.0,
    this.signGamma = 1.0,
    this.micropriceBps = 0.0,
    this.pressureTop5 = 0.0,
    this.regime = '',
    this.regimeCalm = 0.0,
    this.regimeTrend = 0.0,
    this.regimeStress = 0.0,
    this.bands = const {},
    this.detectors = const {},
    this.spectrum = const [],
    this.belief = '',
    this.beliefProbability = 0.0,
    this.runnerUp = '',
    this.certainty = 0.0,
    this.surpriseKl = 0.0,
    this.posterior = const {},
    this.evidence = const [],
    this.chain = const [],
  });

  final bool available;
  final String reason;
  final int ticks;
  final int computeUs;
  final double branchingRatio;
  final double intensityHz;
  final double baselineHz;
  final double vpin;
  final double kyleLambdaBps;
  final double kyleR2;
  final double impactNorm;
  final double signMemory;
  final double signGamma;
  final double micropriceBps;
  final double pressureTop5;
  final String regime;
  final double regimeCalm;
  final double regimeTrend;
  final double regimeStress;
  final Map<String, DeepBand> bands;

  /// ignition / toxicity / stuffing / spoofing / pushable, each 0-1.
  final Map<String, double> detectors;

  /// (scale label, share of energy) from the finest scale up.
  final List<MapEntry<String, double>> spectrum;
  final String belief;
  final double beliefProbability;
  final String runnerUp;
  final double certainty;
  final double surpriseKl;
  final Map<String, double> posterior;

  /// (label, log-odds) - the strongest evidence for the believed emotion.
  final List<MapEntry<String, double>> evidence;
  final List<DeepStep> chain;

  static const DeepReasoning none = DeepReasoning();

  factory DeepReasoning.fromJson(Map<String, dynamic> json) {
    if (json['available'] != true) {
      return DeepReasoning(reason: _s(json['reason']));
    }
    final hawkes = (json['hawkes'] as Map?) ?? const {};
    final flow = (json['flow'] as Map?) ?? const {};
    final book = (json['book'] as Map?) ?? const {};
    final regime = (json['regime'] as Map?) ?? const {};
    final post = (json['posterior'] as Map?) ?? const {};
    final bands = <String, DeepBand>{};
    ((json['bands'] as Map?) ?? const {}).forEach((key, value) {
      if (value is Map) {
        bands[_s(key)] = DeepBand.fromJson(Map<String, dynamic>.from(value));
      }
    });
    final detectors = <String, double>{};
    ((json['manipulation'] as Map?) ?? const {}).forEach((key, value) {
      final name = _s(key);
      if (const ['ignition', 'toxicity', 'stuffing', 'spoofing', 'pushable']
          .contains(name)) {
        detectors[name] = _d(value);
      }
    });
    final posterior = <String, double>{};
    ((post['posterior'] as Map?) ?? const {}).forEach((key, value) {
      posterior[_s(key)] = _d(value);
    });
    final argmax = _s(post['argmax']);
    final evidenceFor = (post['evidence_for'] as Map?) ?? const {};
    final evidence = ((evidenceFor[argmax] as List?) ?? const [])
        .whereType<Map>()
        .map((row) => MapEntry(_s(row['label']), _d(row['log_odds'])))
        .toList();
    return DeepReasoning(
      available: true,
      ticks: _i(json['ticks']),
      computeUs: _i(json['compute_us']),
      branchingRatio: _d(hawkes['branching_ratio']),
      intensityHz: _d(hawkes['intensity_hz']),
      baselineHz: _d(hawkes['baseline_hz']),
      vpin: _d(flow['vpin']),
      kyleLambdaBps: _d(flow['kyle_lambda_bps']),
      kyleR2: _d(flow['kyle_r2']),
      impactNorm: _d(flow['impact_norm']),
      signMemory: _d(flow['sign_memory']),
      signGamma: _d(flow['sign_gamma'], 1.0),
      micropriceBps: _d(book['microprice_bps']),
      pressureTop5: _d(book['pressure_top5']),
      regime: _s(regime['label']),
      regimeCalm: _d(regime['calm']),
      regimeTrend: _d(regime['trend']),
      regimeStress: _d(regime['stress']),
      bands: bands,
      detectors: detectors,
      spectrum: ((json['spectrum'] as List?) ?? const [])
          .whereType<Map>()
          .map((row) => MapEntry(_s(row['scale_label']), _d(row['share'])))
          .toList(),
      belief: argmax,
      beliefProbability: _d(post['argmax_probability']),
      runnerUp: _s(post['runner_up']),
      certainty: _d(post['certainty']),
      surpriseKl: _d(post['surprise_kl']),
      posterior: posterior,
      evidence: evidence,
      chain: ((json['chain'] as List?) ?? const [])
          .whereType<Map>()
          .map((row) => DeepStep.fromJson(Map<String, dynamic>.from(row)))
          .toList(),
    );
  }
}

/// The block every payload carries: the live reading, the reading taken on
/// the frozen snapshot when the signal was locked, and the confidence
/// dampening the crowd produced.
class CrowdEmotions {
  const CrowdEmotions({
    this.live = EmotionReading.none,
    this.locked = EmotionReading.none,
    this.dampeningApplied = 1.0,
    this.dampeningNote = '',
    this.dampenThreshold = 0.45,
    this.dampenMax = 0.25,
  });

  final EmotionReading live;
  final EmotionReading locked;

  /// The multiplier the fusion applied to the confidence (1.0 = none).
  final double dampeningApplied;
  final String dampeningNote;
  final double dampenThreshold;
  final double dampenMax;

  static const CrowdEmotions none = CrowdEmotions();

  bool get hasData => live.hasData;
  bool get dampened => dampeningApplied < 0.999;

  /// From the `emotions` block of a snapshot / PULSE (the full form).
  factory CrowdEmotions.fromJson(Map<String, dynamic> json) {
    final locked = json['locked'];
    final dampening = (json['dampening'] as Map?) ?? const {};
    return CrowdEmotions(
      live: EmotionReading.fromJson(json),
      locked: locked is Map
          ? EmotionReading.fromJson(Map<String, dynamic>.from(locked))
          : EmotionReading.none,
      dampeningApplied: _d(dampening['applied'], 1.0),
      dampeningNote: _s(dampening['note']),
      dampenThreshold: _d(dampening['threshold'], 0.45),
      dampenMax: _d(dampening['max'], 0.25),
    );
  }

  /// A new live reading (the EMOTION stream) keeps the locked one and the
  /// dampening unless the message carries fresher values.
  CrowdEmotions withLive(EmotionReading reading, {Map<String, dynamic>? dampening}) =>
      CrowdEmotions(
        live: reading,
        locked: locked,
        dampeningApplied: _d(dampening?['applied'], dampeningApplied),
        dampeningNote: _s(dampening?['note'], dampeningNote),
        dampenThreshold: dampenThreshold,
        dampenMax: dampenMax,
      );
}

/// Which window the locked signal governs, and what the engine is doing in it.
///
/// The pipeline means `computedAt` is *always* inside the previous countdown:
/// the countdown the user reads shows a signal that already existed when the
/// window started, while the engine computes the next one.
/// The master clock block the backend publishes with every window.
///
/// It is deliberately absolute: the window's start and end are epoch
/// milliseconds, so the countdown is computed against a fixed instant and never
/// restarts or jumps when an unrelated message arrives.  [ticks] are the
/// in-window refresh marks (t+15, t+30, t+45 of a 60-second window) - every
/// panel refreshes on them, together.
class MasterClock {
  const MasterClock({
    this.windowStartedAtMs = 0,
    this.windowEndsAtMs = 0,
    this.serverTimeMs = 0,
    this.cycleId = 0,
    this.periodSeconds = 60.0,
    this.windowSeconds = 60.0,
    this.secondsRemaining = 60.0,
    this.minuteAligned = false,
    this.freshnessMaxAgeSeconds = 65.0,
    this.scoringHorizonSeconds = 60.0,
    this.ticks = const [],
  });

  final int windowStartedAtMs;
  final int windowEndsAtMs;
  final int serverTimeMs;
  final int cycleId;
  final double periodSeconds;
  final double windowSeconds;
  final double secondsRemaining;
  final bool minuteAligned;
  final double freshnessMaxAgeSeconds;
  final double scoringHorizonSeconds;
  final List<ClockTick> ticks;

  /// The next refresh mark, or null at the very end of a window.
  ClockTick? get nextTick {
    for (final tick in ticks) {
      if (!tick.done && tick.secondsUntil > 0) return tick;
    }
    return null;
  }

  factory MasterClock.fromJson(Map<String, dynamic> json) => MasterClock(
        windowStartedAtMs: _i(json['window_started_at_ms']),
        windowEndsAtMs: _i(json['window_ends_at_ms']),
        serverTimeMs: _i(json['server_time_ms']),
        cycleId: _i(json['cycle_id']),
        periodSeconds: _d(json['period_seconds'], 60),
        windowSeconds: _d(json['window_seconds'], 60),
        secondsRemaining: _d(json['seconds_remaining'], 60),
        minuteAligned: json['minute_aligned'] == true,
        freshnessMaxAgeSeconds: _d(json['freshness_max_age_seconds'], 65),
        scoringHorizonSeconds: _d(json['scoring_horizon_seconds'], 60),
        ticks: ((json['ticks'] as List?) ?? const [])
            .map((t) => ClockTick.fromJson(Map<String, dynamic>.from(t as Map)))
            .toList(),
      );
}

/// One refresh mark inside a window.
class ClockTick {
  const ClockTick({
    this.parts = const [],
    this.offsetSeconds = 0.0,
    this.secondsUntil = 0.0,
    this.done = false,
  });

  final List<String> parts;
  final double offsetSeconds;
  final double secondsUntil;
  final bool done;

  factory ClockTick.fromJson(Map<String, dynamic> json) => ClockTick(
        parts: ((json['parts'] as List?) ?? const [])
            .map((p) => p.toString())
            .toList(),
        offsetSeconds: _d(json['offset_seconds']),
        secondsUntil: _d(json['seconds_until']),
        done: json['done'] == true,
      );
}

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
    this.clock,
    this.dataAgeAtOpenSeconds,
    this.agentsLeadSeconds = 8.0,
    this.finalLockLeadSeconds = 0.4,
  });

  /// Round L: the two-stage lock - the agents are prepared
  /// [agentsLeadSeconds] early, the tape / formulas / crowd / fusion are
  /// frozen again [finalLockLeadSeconds] before the boundary, so the window
  /// opened on data [dataAgeAtOpenSeconds] old (null until the first lock).
  final double? dataAgeAtOpenSeconds;
  final double agentsLeadSeconds;
  final double finalLockLeadSeconds;

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

  /// The authoritative clock for this window (null on very old payloads).
  final MasterClock? clock;

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
        clock: json['clock'] is Map
            ? MasterClock.fromJson(
                Map<String, dynamic>.from(json['clock'] as Map))
            : null,
        dataAgeAtOpenSeconds:
            (json['lock'] as Map?)?['data_age_at_open_seconds'] == null
                ? null
                : _d((json['lock'] as Map)['data_age_at_open_seconds']),
        agentsLeadSeconds:
            _d((json['lock'] as Map?)?['agents_lead_seconds'], 8.0),
        finalLockLeadSeconds:
            _d((json['lock'] as Map?)?['final_lock_lead_seconds'], 0.4),
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
    this.units = '',
    this.sensitivity = '',
    this.range = '',
    this.misleads = const [],
    this.corroborates = const [],
  });

  final int index;
  final String name;
  final String title;
  final String category;
  final String brainNode;
  final String description;
  final bool directional;
  final double budgetMs;

  // --- Round I: the logic detail behind the number -------------------------
  /// What the value is measured in ("-1 … +1 share of aggressive volume").
  final String units;

  /// How much market movement one unit of the value represents.
  final String sensitivity;

  /// The band the value normally lives in.
  final String range;

  /// The situations in which this formula's reading is actively misleading.
  final List<String> misleads;

  /// Formulas measuring the same phenomenon from another angle.
  final List<String> corroborates;

  factory FormulaSpec.fromJson(Map<String, dynamic> json) {
    final logic = (json['logic'] as Map?) ?? const {};
    return FormulaSpec(
      index: _i(json['index']),
      name: _s(json['name']),
      title: _s(json['title']),
      category: _s(json['category']),
      brainNode: _s(json['brain_node']),
      description: _s(json['description']),
      directional: json['directional'] != false,
      budgetMs: _d(json['latency_ms']),
      units: _s(logic['units']),
      sensitivity: _s(logic['sensitivity']),
      range: _s(logic['range']),
      misleads: ((logic['misleads'] as List?) ?? const [])
          .map((value) => value.toString())
          .toList(),
      corroborates: ((logic['corroborates'] as List?) ?? const [])
          .map((value) => value.toString())
          .toList(),
    );
  }
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
