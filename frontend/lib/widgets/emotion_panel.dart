import 'dart:ui' show FontFeature;

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
      title: 'Crowd emotion · live · µs → minutes',
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
