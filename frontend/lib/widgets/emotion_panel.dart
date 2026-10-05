
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/signal.dart';
import '../state/app_state.dart';
import '../theme.dart';

/// Crowd Emotion (Round J).
///
/// The user's premise: a 60-second market is easily pushed around by retail
/// emotion.  This panel says which emotion is dominant *right now*, on which
/// timescale (µs → minutes), how long it has held, how crowded / manipulated
/// the minute looks, and what the crowd was feeling when the locked signal was
/// computed.  It is fed by the EMOTION stream (twice a second, on the
/// backend's schedule) and by every PULSE / snapshot; it measures nothing of
/// its own.
class EmotionPanel extends StatelessWidget {
  const EmotionPanel({super.key, required this.state});

  final AppState state;

  static const List<String> _timescales = [
    'micro',
    'seconds',
    'window',
    'minutes',
    'news',
  ];

  static String timescaleLabel(String key) {
    switch (key) {
      case 'micro':
        return 'µs';
      case 'seconds':
        return 'sec';
      case 'window':
        return '60 s';
      case 'minutes':
        return 'min';
      case 'news':
        return 'news';
    }
    return key;
  }

  static Color toneColor(String tone) {
    switch (tone) {
      case 'negative':
        return AppTheme.sell;
      case 'positive':
        return AppTheme.buy;
    }
    return AppTheme.hold;
  }

  static String heldLabel(double seconds) {
    if (seconds >= 60) {
      final minutes = seconds ~/ 60;
      final rest = (seconds - minutes * 60).round();
      return '${minutes}m ${rest}s';
    }
    return '${seconds.toStringAsFixed(1)} s';
  }

  @override
  Widget build(BuildContext context) {
    final crowd = state.crowd;
    final live = crowd.live;
    final dominant = live.dominant;

    if (!live.hasData || dominant == null) {
      return Panel(
        title: 'Crowd emotion · live · µs → minutes',
        trailing: const Text('waiting for the first sample…',
            style: TextStyle(color: AppTheme.textMuted, fontSize: 11)),
        child: Text(
          live.reason.isNotEmpty
              ? live.reason
              : 'the crowd has not been measured yet',
          style: const TextStyle(color: AppTheme.textMuted, fontSize: 12),
        ),
      );
    }

    final color = toneColor(dominant.tone);
    final manipulation = live.manipulation;
    final runner = live.runnerUp;

    return Panel(
      title: 'Crowd emotion · live · a confidence modifier, never a vote',
      trailing: Text(
        '${live.asset.isNotEmpty ? live.asset : state.asset} · every '
        '${live.intervalSeconds.toStringAsFixed(1)} s · ${live.resolutionLabel} · '
        '${live.ticks} ticks',
        style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ---- the dominant emotion, big --------------------------------
          const Text('DOMINANT NOW',
              style: TextStyle(
                  color: AppTheme.textMuted,
                  fontSize: 10,
                  letterSpacing: 1.2,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 2),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 350),
                transitionBuilder: (child, animation) => ScaleTransition(
                  scale: Tween<double>(begin: 0.86, end: 1.0).animate(animation),
                  child: FadeTransition(opacity: animation, child: child),
                ),
                child: Text(
                  dominant.label,
                  key: ValueKey<String>(dominant.name),
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.w800,
                    color: color,
                    height: 1.05,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Text(
                  '${dominant.percent.toStringAsFixed(0)}%',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: color,
                    fontFeatures: const [FontFeature.tabularFigures()],
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '${dominant.family} family · strongest on the '
            '${timescaleLabel(dominant.dominantTimescale)} timescale · '
            'held ${heldLabel(live.heldSeconds)}'
            '${runner != null ? ' · runner-up ${runner.label} ${runner.percent.toStringAsFixed(0)}%' : ''}'
            '${live.churnPerMinute > 0 ? ' · ${live.churnPerMinute} switch${live.churnPerMinute == 1 ? '' : 'es'} this minute' : ''}',
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
          ),
          if (live.read.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(live.read,
                style: const TextStyle(
                    color: AppTheme.textPrimary, fontSize: 12.5, height: 1.4)),
          ],
          if (dominant.drivers.isNotEmpty) ...[
            const SizedBox(height: 6),
            ...dominant.drivers.map(
              (driver) => Text('· $driver',
                  style: const TextStyle(
                      color: AppTheme.textMuted,
                      fontSize: 11,
                      fontFeatures: [FontFeature.tabularFigures()])),
            ),
          ],
          const SizedBox(height: 12),

          // ---- the eight bars, ranked -----------------------------------
          ...live.emotions.map((item) => _EmotionBar(
                item: item,
                dominant: item.name == dominant.name,
              )),
          const SizedBox(height: 8),

          // ---- Round L: the emotion's formula with the live terms ---------
          _EmotionFormulas(emotions: live.emotions, dominantName: dominant.name),
          const SizedBox(height: 12),

          // ---- Round L: the crowd against the 22 formulas ----------------
          _label('VS THE 22 FORMULAS', 'who has the vote'),
          const SizedBox(height: 6),
          _AgreementBlock(agreement: live.formulaAgreement),
          const SizedBox(height: 10),

          // ---- temperature ----------------------------------------------
          _label('TEMPERATURE', 'fear ← 0 → euphoria'),
          const SizedBox(height: 6),
          _ToneGauge(value: live.toneBias),
          const SizedBox(height: 4),
          Text(
            '${live.toneBias >= 0 ? '+' : ''}${live.toneBias.toStringAsFixed(2)} — '
            '${live.toneBias < -0.25 ? 'the crowd is afraid' : live.toneBias > 0.25 ? 'the crowd is chasing' : 'the crowd is neither afraid nor chasing'}',
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
          ),
          const SizedBox(height: 12),

          // ---- manipulation ---------------------------------------------
          _label('MANIPULATION', 'crowding of this minute'),
          const SizedBox(height: 6),
          _Meter(
            value: manipulation.score,
            color: manipulation.score >= 0.45
                ? AppTheme.sell
                : manipulation.score >= 0.25
                    ? AppTheme.warning
                    : AppTheme.hold,
          ),
          const SizedBox(height: 4),
          Text(
            '${(manipulation.score * 100).toStringAsFixed(0)}% ${manipulation.kind}'
            '${crowd.dampened ? ' · confidence x${crowd.dampeningApplied.toStringAsFixed(2)}' : ''}',
            style: const TextStyle(
                color: AppTheme.textPrimary,
                fontSize: 12,
                fontWeight: FontWeight.w600),
          ),
          if (manipulation.note.isNotEmpty)
            Text(manipulation.note,
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
          const SizedBox(height: 12),

          // ---- timescales of the dominant emotion ------------------------
          _label('TIMESCALES', 'where the dominant emotion lives'),
          const SizedBox(height: 6),
          Row(
            children: _timescales.map((key) {
              final value = dominant.byTimescale[key] ?? 0.0;
              final peak = key == dominant.dominantTimescale;
              return Expanded(
                child: Container(
                  margin: const EdgeInsets.only(right: 5),
                  padding: const EdgeInsets.symmetric(vertical: 5),
                  decoration: BoxDecoration(
                    color: AppTheme.background,
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                        color: peak ? color : AppTheme.border,
                        width: peak ? 1.4 : 1),
                  ),
                  child: Column(
                    children: [
                      Text(timescaleLabel(key),
                          style: TextStyle(
                              fontSize: 9.5,
                              color: peak ? color : AppTheme.textMuted)),
                      Text('${(value * 100).toStringAsFixed(0)}%',
                          style: const TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: AppTheme.textPrimary,
                              fontFeatures: [FontFeature.tabularFigures()])),
                    ],
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 12),

          // ---- at lock time ---------------------------------------------
          _label('AT LOCK TIME', 'what shaped this signal'),
          const SizedBox(height: 4),
          _LockedLine(crowd: crowd),
          const SizedBox(height: 14),

          // ---- Round K: the deep reasoning behind the reading -------------
          _label('DEEP REASONING',
              'microsecond → minute formulas · Bayesian filter'),
          const SizedBox(height: 6),
          _DeepBlock(deep: live.deep, dominantName: dominant.name),
        ],
      ),
    );
  }

  static Widget _label(String title, String sub) {
    return RichText(
      text: TextSpan(
        text: title,
        style: const TextStyle(
            color: AppTheme.textMuted,
            fontSize: 10,
            letterSpacing: 1.2,
            fontWeight: FontWeight.w600),
        children: [
          TextSpan(
            text: '  $sub',
            style: const TextStyle(
                letterSpacing: 0, fontWeight: FontWeight.w400, fontSize: 10),
          ),
        ],
      ),
    );
  }
}

class _EmotionBar extends StatelessWidget {
  const _EmotionBar({required this.item, required this.dominant});

  final EmotionScore item;
  final bool dominant;

  @override
  Widget build(BuildContext context) {
    final color = EmotionPanel.toneColor(item.tone);
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          SizedBox(
            width: 94,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(item.label,
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight:
                            dominant ? FontWeight.w800 : FontWeight.w600,
                        color: dominant
                            ? AppTheme.textPrimary
                            : const Color(0xFFCBD7EE))),
                Text(item.family.toUpperCase(),
                    style: const TextStyle(
                        fontSize: 8.5,
                        letterSpacing: .6,
                        color: AppTheme.textMuted)),
              ],
            ),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: Stack(
                children: [
                  Container(height: 8, color: AppTheme.background),
                  AnimatedFractionallySizedBox(
                    duration: const Duration(milliseconds: 350),
                    curve: Curves.easeOut,
                    widthFactor: item.intensity.clamp(0.0, 1.0),
                    child: Container(height: 8, color: color),
                  ),
                ],
              ),
            ),
          ),
          SizedBox(
            width: 40,
            child: Text(
              '${item.percent.toStringAsFixed(0)}%',
              textAlign: TextAlign.right,
              style: const TextStyle(
                  fontSize: 12,
                  color: AppTheme.textPrimary,
                  fontFeatures: [FontFeature.tabularFigures()]),
            ),
          ),
          SizedBox(
            width: 40,
            child: Text(
              EmotionPanel.timescaleLabel(item.dominantTimescale),
              textAlign: TextAlign.right,
              style: const TextStyle(fontSize: 10, color: AppTheme.textMuted),
            ),
          ),
        ],
      ),
    );
  }
}

/// Signed gauge: −1 (fear) … +1 (euphoria), the marker at the crowd's tone.
class _ToneGauge extends StatelessWidget {
  const _ToneGauge({required this.value});

  final double value;

  @override
  Widget build(BuildContext context) {
    final fraction = ((value.clamp(-1.0, 1.0) + 1.0) / 2.0);
    return SizedBox(
      height: 18,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final width = constraints.maxWidth;
          return Stack(
            clipBehavior: Clip.none,
            children: [
              Positioned(
                top: 4,
                left: 0,
                right: 0,
                child: Container(
                  height: 10,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(999),
                    gradient: const LinearGradient(colors: [
                      AppTheme.sell,
                      Color(0xFF33415C),
                      AppTheme.buy,
                    ]),
                  ),
                ),
              ),
              Positioned(
                top: 1,
                left: width / 2,
                child: Container(width: 1, height: 16, color: AppTheme.textMuted),
              ),
              AnimatedPositioned(
                duration: const Duration(milliseconds: 350),
                curve: Curves.easeOut,
                top: 0,
                left: (fraction * width - 9).clamp(0.0, width - 18),
                child: Container(
                  width: 18,
                  height: 18,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.textPrimary,
                    border: Border.all(color: AppTheme.background, width: 2),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _Meter extends StatelessWidget {
  const _Meter({required this.value, required this.color});

  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(999),
      child: Stack(
        children: [
          Container(height: 10, color: AppTheme.background),
          AnimatedFractionallySizedBox(
            duration: const Duration(milliseconds: 350),
            curve: Curves.easeOut,
            widthFactor: value.clamp(0.0, 1.0),
            child: Container(height: 10, color: color),
          ),
        ],
      ),
    );
  }
}

class _LockedLine extends StatelessWidget {
  const _LockedLine({required this.crowd});

  final CrowdEmotions crowd;

  @override
  Widget build(BuildContext context) {
    final locked = crowd.locked;
    final top = locked.dominant;
    if (!locked.hasData || top == null) {
      return const Text(
        'the first locked window will record the crowd\'s mood',
        style: TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '${top.label} ${top.percent.toStringAsFixed(0)}% · manipulation '
          '${(locked.manipulation.score * 100).toStringAsFixed(0)}% '
          '(${locked.manipulation.kind})',
          style: TextStyle(
              color: EmotionPanel.toneColor(top.tone),
              fontSize: 12,
              fontWeight: FontWeight.w600),
        ),
        Text(
          crowd.dampeningNote.isNotEmpty
              ? crowd.dampeningNote
              : 'no crowd dampening on this signal',
          style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
        ),
      ],
    );
  }
}


/// Round K.  The microstructure formulas behind the reading (Hawkes, VPIN,
/// Kyle's lambda, variance ratio / Hurst / entropy per band, sign memory,
/// wavelet spectrum, regime filter, ignition / stuffing / spoofing) and the
/// Bayesian filter's belief, followed by the ordered reasoning chain.  All of
/// it is computed on the server at the emotion cadence; this widget draws it.
class _DeepBlock extends StatelessWidget {
  const _DeepBlock({required this.deep, required this.dominantName});

  final DeepReasoning deep;
  final String dominantName;

  static const _mono = TextStyle(
    fontFamily: 'monospace',
    fontSize: 10.5,
    color: AppTheme.textMuted,
    fontFeatures: [FontFeature.tabularFigures()],
  );

  static String _n(double v, [int d = 2]) => v.toStringAsFixed(d);
  static String _pct(double v) => '${(v * 100).toStringAsFixed(0)}%';
  static String _cap(String s) =>
      s.isEmpty ? '—' : s[0].toUpperCase() + s.substring(1).toLowerCase();

  @override
  Widget build(BuildContext context) {
    if (!deep.available) {
      return Text(
        deep.reason.isNotEmpty
            ? deep.reason
            : 'the deep layer needs a little more tape',
        style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
      );
    }
    final agrees = deep.belief == dominantName;
    final beliefColor = agrees ? AppTheme.buy : AppTheme.warning;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // belief line
        RichText(
          text: TextSpan(
            style: const TextStyle(
                color: AppTheme.textPrimary, fontSize: 12.5, height: 1.4),
            children: [
              const TextSpan(text: 'Bayesian filter: '),
              TextSpan(
                  text: '${_cap(deep.belief)} ${_pct(deep.beliefProbability)}',
                  style: TextStyle(
                      color: beliefColor, fontWeight: FontWeight.w700)),
              TextSpan(
                text: agrees
                    ? '  agrees with the reading'
                    : '  leans differently from the reading',
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
              ),
              TextSpan(
                text: ' · certainty ${_n(deep.certainty)} · surprise '
                    '${_n(deep.surpriseKl)} nats · runner-up '
                    '${_cap(deep.runnerUp)} · ${deep.ticks} ticks in '
                    '${(deep.computeUs / 1000).toStringAsFixed(1)} ms',
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
              ),
            ],
          ),
        ),
        if (deep.evidence.isNotEmpty) ...[
          const SizedBox(height: 4),
          Wrap(
            spacing: 4,
            runSpacing: 4,
            children: deep.evidence
                .take(4)
                .map((e) => _chip(
                    '${e.key} ${e.value >= 0 ? '+' : ''}${_n(e.value)}',
                    e.value >= 0 ? AppTheme.buy : AppTheme.sell))
                .toList(),
          ),
        ],
        const SizedBox(height: 10),

        // verdict tiles
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            _tile('HAWKES n', _n(deep.branchingRatio),
                deep.branchingRatio >= 0.6
                    ? 'cascade'
                    : deep.branchingRatio >= 0.3
                        ? 'clustered'
                        : 'Poisson',
                deep.branchingRatio >= 0.6 ? AppTheme.sell : null),
            _tile('INTENSITY', '${_n(deep.intensityHz, 1)}/s',
                'baseline ${_n(deep.baselineHz, 1)}/s'),
            _tile(
                'VPIN',
                _n(deep.vpin),
                deep.vpin >= 0.6
                    ? 'one-sided'
                    : deep.vpin >= 0.4
                        ? 'leaning'
                        : 'balanced',
                deep.vpin >= 0.6 ? AppTheme.sell : null),
            _tile('KYLE λ', _n(deep.kyleLambdaBps, 4),
                'R² ${_n(deep.kyleR2)} · ${_n(deep.impactNorm)}x'),
            _tile('SIGN MEMORY', _n(deep.signMemory),
                'γ ${_n(deep.signGamma)}'),
            _tile(
                'MICROPRICE',
                '${deep.micropriceBps >= 0 ? '+' : ''}${_n(deep.micropriceBps, 3)} bps',
                'top-5 ${deep.pressureTop5 >= 0 ? '+' : ''}${_n(deep.pressureTop5)}'),
            _tile(
                'REGIME',
                deep.regime,
                'calm ${_pct(deep.regimeCalm)} · trend ${_pct(deep.regimeTrend)}'
                ' · stress ${_pct(deep.regimeStress)}',
                deep.regime == 'stress'
                    ? AppTheme.sell
                    : deep.regime == 'calm'
                        ? AppTheme.buy
                        : null),
          ],
        ),
        const SizedBox(height: 10),

        // bands table
        _bandRow('band', 'clock', 'H', 'VR', 'entropy', header: true),
        for (final key in const ['micro', 'seconds', 'window'])
          if (deep.bands[key] != null)
            _bandRow(
              key,
              deep.bands[key]!.clock,
              _n(deep.bands[key]!.hurst),
              _n(deep.bands[key]!.varianceRatio),
              _n(deep.bands[key]!.entropy),
              hurst: deep.bands[key]!.hurst,
              vr: deep.bands[key]!.varianceRatio,
            ),
        const SizedBox(height: 10),

        // detectors
        for (final entry in const [
          MapEntry('ignition', 'momentum ignition'),
          MapEntry('toxicity', 'toxic flow'),
          MapEntry('stuffing', 'quote stuffing'),
          MapEntry('spoofing', 'spoofing'),
          MapEntry('pushable', 'pushable tape'),
        ])
          Padding(
            padding: const EdgeInsets.only(bottom: 3),
            child: Row(
              children: [
                SizedBox(
                  width: 120,
                  child: Text(entry.value,
                      style: const TextStyle(
                          color: AppTheme.textMuted, fontSize: 11)),
                ),
                Expanded(
                  child: _Meter(
                    value: deep.detectors[entry.key] ?? 0.0,
                    color: (deep.detectors[entry.key] ?? 0.0) >= 0.5
                        ? AppTheme.sell
                        : (deep.detectors[entry.key] ?? 0.0) >= 0.25
                            ? AppTheme.warning
                            : AppTheme.hold,
                  ),
                ),
                SizedBox(
                  width: 40,
                  child: Text(_pct(deep.detectors[entry.key] ?? 0.0),
                      textAlign: TextAlign.right,
                      style: const TextStyle(fontSize: 11)),
                ),
              ],
            ),
          ),
        // ---- Round Z geometry + Round AA inverse RL ---------------------
        if (deep.geometry.available) ...[
          const SizedBox(height: 10),
          _GeometryBlock(geometry: deep.geometry),
        ],
        if (deep.utility.available) ...[
          const SizedBox(height: 10),
          _UtilityBlock(utility: deep.utility),
        ],
        if (deep.spectrum.isNotEmpty) ...[
          const SizedBox(height: 8),
          Wrap(
            spacing: 4,
            runSpacing: 4,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              const Text('where the energy is',
                  style: TextStyle(color: AppTheme.textMuted, fontSize: 11)),
              ...deep.spectrum.map((sc) {
                final peak = sc.value ==
                    deep.spectrum
                        .map((e) => e.value)
                        .reduce((a, b) => a > b ? a : b);
                return Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                  decoration: BoxDecoration(
                    border: Border.all(
                        color: peak ? AppTheme.accent : AppTheme.border),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Column(
                    children: [
                      Text(sc.key,
                          style: TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.w600,
                              color: peak
                                  ? AppTheme.accent
                                  : AppTheme.textPrimary)),
                      Text(_pct(sc.value),
                          style: const TextStyle(fontSize: 10.5)),
                    ],
                  ),
                );
              }),
            ],
          ),
        ],
        const SizedBox(height: 10),

        // the chain
        Theme(
          data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
          child: ExpansionTile(
            tilePadding: EdgeInsets.zero,
            childrenPadding: EdgeInsets.zero,
            initiallyExpanded: true,
            title: Text(
              'REASONING CHAIN  ${deep.chain.length} steps, microseconds → minute',
              style: const TextStyle(
                  color: AppTheme.textMuted,
                  fontSize: 10,
                  letterSpacing: 1.2,
                  fontWeight: FontWeight.w600),
            ),
            children: deep.chain.map(_stepTile).toList(),
          ),
        ),
      ],
    );
  }

  Widget _stepTile(DeepStep step) {
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.fromLTRB(8, 6, 8, 6),
      decoration: BoxDecoration(
        border: Border.all(color: AppTheme.border),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 18,
            height: 18,
            margin: const EdgeInsets.only(right: 8, top: 1),
            alignment: Alignment.center,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: AppTheme.accent),
            ),
            child: Text('${step.step}',
                style: const TextStyle(fontSize: 10, color: AppTheme.accent)),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(step.name,
                          style: const TextStyle(
                              fontSize: 11.5, fontWeight: FontWeight.w700)),
                    ),
                    Text('${step.value} ${step.unit}',
                        style: _mono.copyWith(color: AppTheme.textPrimary)),
                    const SizedBox(width: 6),
                    Text(step.timescale,
                        style: const TextStyle(
                            fontSize: 10, color: AppTheme.textMuted)),
                  ],
                ),
                Text(step.formula, style: _mono),
                Wrap(
                  spacing: 4,
                  runSpacing: 2,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    Text(step.reads,
                        style: const TextStyle(fontSize: 11, height: 1.4)),
                    if (step.feeds.isNotEmpty)
                      const Text('→',
                          style: TextStyle(
                              fontSize: 11, color: AppTheme.textMuted)),
                    ...step.feeds.map((f) => _chip(f, AppTheme.textMuted)),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  static Widget _chip(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 1),
      decoration: BoxDecoration(
        border: Border.all(color: color.withAlpha(140)),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(text, style: TextStyle(fontSize: 10.5, color: color)),
    );
  }

  static Widget _tile(String label, String value, String sub, [Color? color]) {
    return Container(
      width: 150,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        border: Border.all(color: AppTheme.border),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: const TextStyle(
                  color: AppTheme.textMuted, fontSize: 9.5, letterSpacing: 1)),
          Text(value,
              style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: color ?? AppTheme.textPrimary,
                  fontFeatures: const [FontFeature.tabularFigures()])),
          Text(sub,
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
        ],
      ),
    );
  }

  static Widget _bandRow(String a, String b, String c, String d, String e,
      {bool header = false, double? hurst, double? vr}) {
    Color? tint(double? v, double hi, double lo) {
      if (v == null) return null;
      if (v > hi) return AppTheme.buy;
      if (v < lo) return AppTheme.sell;
      return null;
    }

    final style = TextStyle(
      fontSize: header ? 9.5 : 11,
      letterSpacing: header ? 1 : 0,
      color: header ? AppTheme.textMuted : AppTheme.textPrimary,
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 3),
      decoration: const BoxDecoration(
          border: Border(bottom: BorderSide(color: AppTheme.border))),
      child: Row(
        children: [
          Expanded(flex: 11, child: Text(header ? a.toUpperCase() : a, style: style)),
          Expanded(flex: 10, child: Text(header ? b.toUpperCase() : b, style: style)),
          Expanded(
              flex: 8,
              child: Text(c, style: style.copyWith(color: tint(hurst, 0.58, 0.42) ?? style.color))),
          Expanded(
              flex: 8,
              child: Text(d, style: style.copyWith(color: tint(vr, 1.15, 0.85) ?? style.color))),
          Expanded(flex: 9, child: Text(e, style: style)),
        ],
      ),
    );
  }
}


/// Round L: the eight emotions are printed formulas - weighted sums of
/// bounded ramps over the tape's own surprise units, times a gate.  This block
/// prints the selected emotion's formula with every term's weight, live value
/// and contribution.  Tap a chip to switch emotion; the dominant one is shown
/// by default.
class _EmotionFormulas extends StatefulWidget {
  const _EmotionFormulas({required this.emotions, required this.dominantName});

  final List<EmotionScore> emotions;
  final String dominantName;

  @override
  State<_EmotionFormulas> createState() => _EmotionFormulasState();
}

class _EmotionFormulasState extends State<_EmotionFormulas> {
  String? _selected;

  @override
  Widget build(BuildContext context) {
    final name = _selected ?? widget.dominantName;
    EmotionScore? item;
    for (final row in widget.emotions) {
      if (row.name == name) item = row;
    }
    item ??= widget.emotions.isNotEmpty ? widget.emotions.first : null;
    if (item == null || item.formula.isEmpty) return const SizedBox.shrink();
    final raw = item.terms.fold<double>(0.0, (acc, t) => acc + t.contribution);
    const mono = TextStyle(
        color: AppTheme.textMuted,
        fontSize: 10.5,
        fontFamily: 'mono',
        fontFeatures: [FontFeature.tabularFigures()]);
    const monoStrong = TextStyle(
        color: AppTheme.textPrimary,
        fontSize: 10.5,
        fontFamily: 'mono',
        fontFeatures: [FontFeature.tabularFigures()]);
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0x08FFFFFF),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 6,
            runSpacing: 4,
            children: widget.emotions
                .map((row) => ChoiceChip(
                      label: Text(row.label,
                          style: const TextStyle(fontSize: 10.5)),
                      selected: row.name == item!.name,
                      visualDensity: VisualDensity.compact,
                      onSelected: (_) => setState(() => _selected = row.name),
                    ))
                .toList(),
          ),
          const SizedBox(height: 8),
          Text('${item.label} · formula · live terms',
              style: const TextStyle(
                  color: AppTheme.textPrimary,
                  fontSize: 11.5,
                  fontWeight: FontWeight.w700)),
          const SizedBox(height: 4),
          SelectableText(item.formula,
              style: const TextStyle(
                  color: AppTheme.textPrimary,
                  fontSize: 10.5,
                  fontFamily: 'mono',
                  height: 1.4)),
          const SizedBox(height: 6),
          Table(
            columnWidths: const {
              0: FixedColumnWidth(40),
              1: FlexColumnWidth(),
              2: FixedColumnWidth(52),
              3: FixedColumnWidth(56),
            },
            children: [
              const TableRow(children: [
                Text('w', style: mono),
                Text('term', style: mono),
                Text('value', style: mono, textAlign: TextAlign.right),
                Text('w·value', style: mono, textAlign: TextAlign.right),
              ]),
              ...item.terms.map((t) => TableRow(children: [
                    Text(t.weight.toStringAsFixed(2), style: monoStrong),
                    Text(t.term, style: mono),
                    Text(t.value.toStringAsFixed(3),
                        style: monoStrong, textAlign: TextAlign.right),
                    Text(t.contribution.toStringAsFixed(3),
                        style: monoStrong, textAlign: TextAlign.right),
                  ])),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            'Σ = ${raw.toStringAsFixed(3)} × gate ${item.gate.toStringAsFixed(3)} '
            '→ ramp ${item.ramp.toStringAsFixed(3)} · Bayesian belief '
            '${(item.belief * 100).toStringAsFixed(0)}% · shown '
            '${item.percent.toStringAsFixed(0)}% '
            '(0.65·ramp + 0.35·min(1, 2.5·belief), band-lifted, EMA-smoothed)',
            style: mono,
          ),
        ],
      ),
    );
  }
}

/// Round L: the crowd against the 22 formulas.  The formulas carry the vote
/// (40% of the fusion, the largest weight); the crowd is a bounded confidence
/// modifier.  The block says whether the two agree and repeats the rule.
class _AgreementBlock extends StatelessWidget {
  const _AgreementBlock({required this.agreement});

  final FormulaAgreement agreement;

  @override
  Widget build(BuildContext context) {
    if (!agreement.hasVerdict) {
      return const Text('waiting for the first formula pass…',
          style: TextStyle(color: AppTheme.textMuted, fontSize: 11));
    }
    final color = agreement.verdict == 'aligned'
        ? AppTheme.buy
        : agreement.verdict == 'conflict'
            ? AppTheme.sell
            : AppTheme.textMuted;
    final names = [
      ...agreement.upNames.map((n) => '$n↑'),
      ...agreement.downNames.map((n) => '$n↓'),
    ].take(6).join(' ');
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(agreement.verdict.toUpperCase(),
            style: TextStyle(
                color: color,
                fontSize: 12,
                letterSpacing: 1.1,
                fontWeight: FontWeight.w800)),
        const SizedBox(height: 6),
        _gaugeRow('formulas', agreement.consensus),
        const SizedBox(height: 4),
        _gaugeRow('crowd', agreement.crowdTone),
        const SizedBox(height: 6),
        Text(agreement.note,
            style: const TextStyle(
                color: AppTheme.textPrimary, fontSize: 11.5, height: 1.4)),
        if (names.isNotEmpty)
          Text(names,
              style: const TextStyle(
                  color: AppTheme.textMuted, fontSize: 10.5, fontFamily: 'mono')),
        const SizedBox(height: 2),
        Text(agreement.rule,
            style: const TextStyle(
                color: AppTheme.textMuted, fontSize: 10.5, height: 1.4)),
      ],
    );
  }

  Widget _gaugeRow(String label, double value) {
    return Row(
      children: [
        SizedBox(
            width: 58,
            child: Text(label,
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11))),
        Expanded(child: _ToneGauge(value: value)),
        const SizedBox(width: 8),
        SizedBox(
          width: 48,
          child: Text(
            '${value >= 0 ? '+' : ''}${value.toStringAsFixed(2)}',
            textAlign: TextAlign.right,
            style: const TextStyle(
                color: AppTheme.textPrimary,
                fontSize: 11,
                fontFamily: 'mono',
                fontFeatures: [FontFeature.tabularFigures()]),
          ),
        ),
      ],
    );
  }
}


/// GEOMETRY OF THE CROWD - the five measurements of `deep.geometry`
/// (book topology, Takens attractor, critical slowing down, entropy
/// production, decision interference), each as an intensity meter with its
/// headline number, mirroring the dashboard tiles.
class _GeometryBlock extends StatelessWidget {
  const _GeometryBlock({required this.geometry});

  final CrowdGeometry geometry;

  static String _n(double v, [int d = 2]) => v.toStringAsFixed(d);

  @override
  Widget build(BuildContext context) {
    final rows = <MapEntry<String, GeometryMeasure>>[
      MapEntry('TDA · book manifold', geometry.tda),
      MapEntry('Takens · attractor λ', geometry.takens),
      MapEntry('critical slowing', geometry.csd),
      MapEntry('entropy production', geometry.thermo),
      MapEntry('quantum interference', geometry.quantum),
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text('GEOMETRY OF THE CROWD',
                style: TextStyle(
                    color: AppTheme.textMuted,
                    fontSize: 10,
                    letterSpacing: 1.4,
                    fontWeight: FontWeight.w600)),
            const SizedBox(width: 8),
            Text(
                '${geometry.liveMeasurements}/5 live · stress '
                '${_n(geometry.stress)}',
                style:
                    const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
          ],
        ),
        const SizedBox(height: 6),
        for (final row in rows)
          if (row.value.available)
            Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(
                children: [
                  SizedBox(
                    width: 136,
                    child: Text(row.key,
                        style: const TextStyle(
                            color: AppTheme.textMuted, fontSize: 11)),
                  ),
                  Expanded(
                    child: _Meter(
                      value: row.value.intensity.clamp(0.0, 1.0),
                      color: row.value.intensity >= 0.6
                          ? AppTheme.sell
                          : row.value.intensity >= 0.3
                              ? AppTheme.warning
                              : AppTheme.accent,
                    ),
                  ),
                  SizedBox(
                    width: 52,
                    child: Text(_n(row.value.value, 3),
                        textAlign: TextAlign.right,
                        style: const TextStyle(
                            fontSize: 11,
                            fontFeatures: [FontFeature.tabularFigures()])),
                  ),
                  const SizedBox(width: 6),
                  SizedBox(
                    width: 110,
                    child: Text(row.value.detail,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                            color: AppTheme.textMuted, fontSize: 10)),
                  ),
                ],
              ),
            ),
        if (geometry.read.isNotEmpty)
          Text(geometry.read,
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
      ],
    );
  }
}

/// WHAT THE CROWD IS MAXIMISING - the inverse-RL parameters of
/// `deep.utility`: loss aversion λ, risk aversion γ and the probability
/// weighting α, each with its one-line reading and a position on its scale.
class _UtilityBlock extends StatelessWidget {
  const _UtilityBlock({required this.utility});

  final CrowdUtility utility;

  static String _n(double v, [int d = 2]) => v.toStringAsFixed(d);

  static double _pos(double v, double lo, double hi) =>
      ((v - lo) / (hi - lo)).clamp(0.0, 1.0);

  Widget _tile(String label, String value, String sub, double pos,
      {bool available = true, Color? color}) {
    return Container(
      width: 170,
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 7),
      decoration: BoxDecoration(
        border: Border.all(color: AppTheme.border),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: const TextStyle(
                  color: AppTheme.textMuted, fontSize: 9.5, letterSpacing: 1)),
          Text(available ? value : '—',
              style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: color ?? AppTheme.textPrimary,
                  fontFeatures: const [FontFeature.tabularFigures()])),
          Text(available ? sub : 'needs more tape',
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10)),
          const SizedBox(height: 5),
          LayoutBuilder(builder: (context, constraints) {
            final width = constraints.maxWidth;
            return SizedBox(
              height: 10,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  Positioned(
                    left: 0,
                    right: 0,
                    top: 3,
                    child: Container(
                      height: 4,
                      decoration: BoxDecoration(
                        color: AppTheme.border,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  if (available)
                    AnimatedPositioned(
                      duration: const Duration(milliseconds: 600),
                      curve: Curves.easeOutCubic,
                      left: (width * pos - 5).clamp(0.0, width - 10),
                      top: 0,
                      child: Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: color ?? AppTheme.accent,
                          boxShadow: [
                            BoxShadow(
                                color: (color ?? AppTheme.accent).withAlpha(140),
                                blurRadius: 8),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final u = utility;
    final lamColor = u.lambda > 2.5
        ? AppTheme.sell
        : u.lambda < 0.5
            ? AppTheme.buy
            : null;
    final gamColor = u.gamma > 0.3
        ? AppTheme.warning
        : u.gamma < -0.3
            ? AppTheme.buy
            : null;
    final alpColor = u.alpha < 0.85
        ? AppTheme.sell
        : u.alpha > 1.25
            ? AppTheme.warning
            : null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text('WHAT THE CROWD IS MAXIMISING',
                style: TextStyle(
                    color: AppTheme.textMuted,
                    fontSize: 10,
                    letterSpacing: 1.4,
                    fontWeight: FontWeight.w600)),
            const SizedBox(width: 8),
            Text('inverse RL · ${u.liveMeasurements}/3 live',
                style:
                    const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
          ],
        ),
        const SizedBox(height: 6),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _tile(
              'LOSS AVERSION λ',
              _n(u.lambda),
              '${u.lambdaRead} · ${u.lambdaBuckets} s',
              _pos(u.lambda > 0 ? math.log(u.lambda) : 0.0, -1.5, 1.5),
              available: u.lambdaAvailable,
              color: lamColor,
            ),
            _tile(
              'RISK AVERSION γ',
              _n(u.gamma),
              u.gammaRead,
              _pos(u.gamma, -2.0, 2.0),
              available: u.gammaAvailable,
              color: gamColor,
            ),
            _tile(
              'PROBABILITY WEIGHTING α',
              _n(u.alpha),
              '${u.alphaRead} · tail ${(u.tailShare * 100).toStringAsFixed(1)}%',
              _pos(u.alpha, 0.2, 2.0),
              available: u.alphaAvailable,
              color: alpColor,
            ),
          ],
        ),
        if (u.read.isNotEmpty) ...[
          const SizedBox(height: 6),
          Text(u.read,
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
        ],
      ],
    );
  }
}
