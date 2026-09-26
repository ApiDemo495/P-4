import 'dart:math' as math;
import 'dart:ui' show FontFeature;

import 'package:flutter/material.dart';

import '../models/signal.dart';
import '../state/app_state.dart';
import '../theme.dart';

/// The fixed-layout signal widget (feedback round, items 2, 3, 4 and 5).
///
///     row 1:  [ prediction + reasoning ] [ countdown 1-15 ] [ side & conviction ]
///     row 2:  [ take profit & stop loss ]      [ prediction accuracy ]
///
/// Two rules are structural, not cosmetic:
///
/// * the countdown shows a signal that was **already computed** when the
///   window began (the engine is computing the next one while this one runs),
///   so the panel is never blank and the user never waits;
/// * the emergency notice is a small glittering box **inside** the page - there
///   is no full-screen overlay anywhere in the client.
class SignalWidgetPanel extends StatelessWidget {
  const SignalWidgetPanel({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final narrow = constraints.maxWidth < 760;
        // ``fill`` keeps every cell the same height in the wide (widget) layout
        // and lets it size to its content on a phone, where a fixed height
        // would clip the reasoning list.
        final fill = !narrow;
        final top = narrow
            ? Column(
                children: [
                  _predictionCell(fill),
                  const SizedBox(height: 10),
                  _countdownCell(fill),
                  const SizedBox(height: 10),
                  _holdCell(fill),
                ],
              )
            : Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(flex: 105, child: _predictionCell(fill)),
                  const SizedBox(width: 10),
                  Expanded(flex: 100, child: _countdownCell(fill)),
                  const SizedBox(width: 10),
                  Expanded(flex: 115, child: _holdCell(fill)),
                ],
              );

        final bottom = narrow
            ? Column(
                children: [
                  _riskCell(fill),
                  const SizedBox(height: 10),
                  _accuracyCell(fill),
                ],
              )
            : Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(child: _riskCell(fill)),
                  const SizedBox(width: 10),
                  Expanded(child: _accuracyCell(fill)),
                ],
              );

        return Panel(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              top,
              const SizedBox(height: 10),
              bottom,
              const SizedBox(height: 10),
              _strip(),
            ],
          ),
        );
      },
    );
  }

  // =========================================================================
  // Row 1
  // =========================================================================
  Widget _predictionCell(bool fill) {
    final signal = state.signal;
    // NOTE: the null checks are written inline on purpose - Dart's flow
    // analysis promotes `signal` only when the check is in the same expression.
    final value = signal == null ? '· · ·' : signal.signal;
    final color =
        signal == null ? AppTheme.textMuted : AppTheme.forSignal(signal.signal);
    final pending = signal == null;

    return _Cell(
      fill: fill,
      label: 'PREDICTION',
      trailing:
          signal == null ? 'starting up' : 'window #${signal.cycleNumber}',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // AnimatedSwitcher + a scale transition: the direction change is the
          // single most important event on this screen, so it moves.
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 380),
            transitionBuilder: (child, animation) => ScaleTransition(
              scale: Tween<double>(begin: 0.86, end: 1).animate(
                CurvedAnimation(parent: animation, curve: Curves.easeOutBack),
              ),
              child: FadeTransition(opacity: animation, child: child),
            ),
            child: Text(
              value,
              key: ValueKey<String>('prediction-${value}'),
              style: TextStyle(
                color: color,
                fontSize: 44,
                fontWeight: FontWeight.w800,
                letterSpacing: 1.5,
                height: 1.0,
                shadows: pending
                    ? null
                    : [Shadow(color: color.withAlpha(90), blurRadius: 22)],
              ),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Text(
                _predictionSubtitle(signal),
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
              ),
              const SizedBox(width: 6),
              _FreshnessChip(state: state),
            ],
          ),
          const SizedBox(height: 6),
          _ReasonList(state: state),
          // Inline emergency chip: small, glittering, inside the page.
          if (state.emergency != null || (signal?.isEmergencyOverride ?? false))
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: _EmergencyChip(state: state),
            ),
        ],
      ),
    );
  }

  String _predictionSubtitle(FrozenSignal? signal) {
    if (signal == null) return 'computing the first window…';
    if (signal.preview) return 'bootstrap window — computed at the boundary';
    final computed = signal.computedAt.split('T').last.replaceAll('Z', '');
    return 'computed ${computed.isEmpty ? '—' : computed}Z · '
        'confidence ${(signal.confidence * 100).toStringAsFixed(0)}%';
  }

  Widget _countdownCell(bool fill) {
    final period = state.cyclePeriodSeconds <= 0 ? 60.0 : state.cyclePeriodSeconds;
    final remaining = state.secondsRemaining.clamp(0.0, period);
    final seconds = remaining.ceil();
    final progress = (remaining / period).clamp(0.0, 1.0);
    final ready = state.nextWindowReady;
    final w = state.window;

    return _Cell(
      fill: fill,
      label: 'COUNTDOWN',
      trailing: 'seconds to the boundary',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          TweenAnimationBuilder<double>(
            tween: Tween<double>(begin: progress, end: progress),
            duration: const Duration(milliseconds: 220),
            builder: (context, value, _) => SizedBox(
              width: 104,
              height: 104,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  CustomPaint(
                    size: const Size(104, 104),
                    painter: _RingPainter(
                      fraction: value,
                      color: ready ? AppTheme.buy : AppTheme.accent,
                      urgent: seconds <= 5,
                    ),
                  ),
                  Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        '$seconds',
                        style: const TextStyle(
                          fontSize: 34,
                          fontWeight: FontWeight.w700,
                          height: 1.0,
                          fontFeatures: [FontFeature.tabularFigures()],
                        ),
                      ),
                      const Text(
                        'SEC',
                        style: TextStyle(
                          color: AppTheme.textMuted,
                          fontSize: 9,
                          letterSpacing: 1.6,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),
          // The same schedule the backend runs on: when the next refresh mark
          // is, and what it refreshes.  Every panel moves on that tick.
          Text(
            state.countdownNote,
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: state.nextComputeProgress,
              minHeight: 5,
              backgroundColor: AppTheme.background,
              valueColor: AlwaysStoppedAnimation<Color>(
                ready ? AppTheme.buy : AppTheme.accent,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            ready
                ? 'next signal #${state.cycleNumber + 1} ready — revealed at the boundary'
                : 'computing signal #${state.cycleNumber + 1} … '
                    '${(state.nextComputeProgress * 100).round()}%',
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
          ),
          if (w?.computedSecondsAgo != null)
            Text(
              'this window was computed '
              '${w!.computedSecondsAgo!.round()}s before it opened',
              textAlign: TextAlign.center,
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10),
            ),
        ],
      ),
    );
  }

  Widget _holdCell(bool fill) {
    final signal = state.signal;
    final emergency = state.emergency != null || (signal?.isEmergencyOverride ?? false);
    final conviction = signal?.prediction.conviction ??
        (signal == null ? '—' : (signal.confidence > 0.7 ? 'HIGH' : 'MEDIUM'));
    final note = state.holdWarning;
    return _Cell(
      fill: fill,
      label: 'SIGNAL & CONVICTION',
      child: GlitterBox(
        active: emergency,
        emergency: emergency,
        title: emergency
            ? '\u{26A1} EXIT \u{2192} ${signal?.signal ?? '—'}'
            : signal == null
                ? '…'
                : '${signal.signal} · $conviction',
        body: emergency
            ? 'Emergency override: exit any open position now.'
            : signal == null
                ? 'waiting for the first window'
                : (note?.text ??
                    '${signal.signal} is live (${conviction.toLowerCase()} '
                        'conviction) — take profit and stop loss are in the '
                        'row below.'),
        meta: emergency
            ? (state.emergency?['headline']?.toString() ?? signal?.emergencyHeadline ?? '')
            : signal == null
                ? ''
                : 'confidence ${(signal.confidence * 100).toStringAsFixed(0)}% · '
                    '1:1 levels · updated ${(state.prediction.ageSeconds).toStringAsFixed(0)}s ago',
      ),
    );
  }

  // =========================================================================
  // Row 2
  // =========================================================================
  Widget _riskCell(bool fill) {
    final risk = state.signal?.risk ?? SignalRisk.none;
    final entry = risk.entry > 0 ? risk.entry : (state.signal?.price ?? 0);
    final tradeable = risk.tradeable;
    return _Cell(
      fill: fill,
      label: 'TAKE PROFIT / STOP LOSS',
      child: _KvGrid(items: [
        _Kv('Entry', _money(entry)),
        _Kv(
          'Take profit',
          tradeable ? _money(risk.takeProfit) : '—',
          color: tradeable ? AppTheme.buy : AppTheme.textMuted,
        ),
        _Kv(
          'Stop loss',
          tradeable ? _money(risk.stopLoss) : '—',
          color: tradeable ? AppTheme.sell : AppTheme.textMuted,
        ),
        _Kv(
          'Reward : risk (1:1)',
          risk.rr > 0 ? '${risk.rr.toStringAsFixed(2)} : 1' : '—',
        ),
        _Kv(
          'Realised vol · 1 min',
          risk.volatilityBps > 0
              ? '${risk.volatilityBps.toStringAsFixed(1)} bps'
              : '—',
        ),
        _Kv(
          'Levels set',
          tradeable
              ? '${risk.tpBps.round()} / ${risk.slBps.round()} bps'
              : 'no position',
          color: tradeable ? AppTheme.textPrimary : AppTheme.textMuted,
        ),
        _Kv(
          'Note',
          risk.note.isEmpty ? '—' : risk.note,
          wide: true,
          small: true,
        ),
      ]),
    );
  }

  Widget _accuracyCell(bool fill) {
    final streak = state.outcomeStreak;
    final last = state.lastOutcome;
    return _Cell(
      fill: fill,
      label: 'PREDICTION ACCURACY',
      child: _KvGrid(items: [
        _Kv('Win rate', '${(state.winRate * 100).toStringAsFixed(0)}%'),
        _Kv('Evaluated windows', '${state.outcomeCount}'),
        _Kv('BUY hit rate', _sideRate(state.buyAccuracy)),
        _Kv('SELL hit rate', _sideRate(state.sellAccuracy)),
        _Kv(
          'Prediction age',
          state.prediction.ageSeconds == null
              ? '—'
              : '${state.predictionAgeSeconds.toStringAsFixed(0)}s '
                  '/ ${state.prediction.maxAgeSeconds.toStringAsFixed(0)}s',
          color: state.predictionStale ? AppTheme.sell : AppTheme.textMuted,
          small: true,
        ),
        _Kv(
          'Scored after',
          '${state.prediction.horizonSeconds.toStringAsFixed(0)}s',
          small: true,
        ),
        _Kv(
          'Current streak',
          streak == 0
              ? '—'
              : streak > 0
                  ? '$streak win${streak > 1 ? 's' : ''}'
                  : '${-streak} loss${streak < -1 ? 'es' : ''}',
          color: streak > 0
              ? AppTheme.buy
              : streak < 0
                  ? AppTheme.sell
                  : AppTheme.textMuted,
        ),
        _Kv(
          'Last outcome',
          last == null
              ? 'waiting for the first window to close'
              : '${last.label} ${last.pnlBps.toStringAsFixed(1)} bps',
          color: last == null
              ? AppTheme.textMuted
              : last.outcome > 0
                  ? AppTheme.buy
                  : last.outcome < 0
                      ? AppTheme.sell
                      : AppTheme.textMuted,
        ),
        _Kv(
          'Engine',
          '${state.connected ? 'live' : 'reconnecting'} · '
              'L${state.degradationLevel} ${state.degradationLabel} · '
              '${state.cyclePeriodSeconds.round()}s windows'
              '${(state.window?.pipeline ?? true) ? ' · pipelined' : ''}',
          wide: true,
          small: true,
        ),
      ]),
    );
  }

  Widget _strip() {
    final w = state.window;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppTheme.background,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.border),
      ),
      child: Wrap(
        spacing: 18,
        runSpacing: 6,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          _stripItem('cycle', '${state.cycleNumber}'),
          _stripItem('state', state.lockState),
          _stripItem('window', w == null ? '—' : '${_clock(w.validFrom)}–${_clock(w.validUntil)}Z'),
          _stripItem('utc', state.ntpSynced ? 'NTP synced' : 'system clock'),
          Text(
            (w?.pipeline ?? true)
                ? 'pipelined: each window was computed during the previous one'
                : 'SIGNAL_PIPELINE=0 · literal 8-second computing window',
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
          ),
        ],
      ),
    );
  }

  Widget _stripItem(String label, String value) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('$label ',
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
        Text(value,
            style: const TextStyle(
              fontSize: 11.5,
              fontWeight: FontWeight.w600,
              fontFeatures: [FontFeature.tabularFigures()],
            )),
      ],
    );
  }

  static String _clock(String iso) {
    final parts = iso.split('T');
    return parts.length < 2 ? '—' : parts.last.replaceAll('Z', '');
  }

  static String _money(double? value) {
    if (value == null || value <= 0) return '—';
    final fixed = value.toStringAsFixed(2);
    final parts = fixed.split('.');
    final whole = parts.first;
    final buffer = StringBuffer();
    for (var i = 0; i < whole.length; i++) {
      if (i > 0 && (whole.length - i) % 3 == 0) buffer.write(',');
      buffer.write(whole[i]);
    }
    return '\$$buffer.${parts.last}';
  }
}

// ===========================================================================
// Building blocks
// ===========================================================================

class _Cell extends StatelessWidget {
  const _Cell({
    required this.label,
    required this.child,
    this.trailing,
    this.fill = true,
  });

  final String label;
  final Widget child;
  final String? trailing;

  /// ``true`` -> the cell stretches to the row height (the widget layout).
  final bool fill;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF121A29), Color(0xFF0E1522)],
        ),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 10,
                  letterSpacing: 1.5,
                  color: AppTheme.textMuted,
                  fontWeight: FontWeight.w600,
                ),
              ),
              if (trailing != null)
                Flexible(
                  child: Text(
                    trailing!,
                    textAlign: TextAlign.right,
                    style: const TextStyle(color: AppTheme.textMuted, fontSize: 10),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          if (fill) Expanded(child: child) else child,
        ],
      ),
    );
  }
}

/// A small glittering box. Never a full-screen overlay: it lives inside the
/// page, directly under the prediction (item 2).
class GlitterBox extends StatefulWidget {
  const GlitterBox({
    super.key,
    required this.active,
    required this.title,
    required this.body,
    required this.meta,
    this.emergency = false,
  });

  final bool active;
  final bool emergency;
  final String title;
  final String body;
  final String meta;

  @override
  State<GlitterBox> createState() => _GlitterBoxState();
}

class _GlitterBoxState extends State<GlitterBox>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: widget.emergency
        ? const Duration(milliseconds: 1500)
        : const Duration(milliseconds: 3400),
  );

  @override
  void initState() {
    super.initState();
    if (widget.active) _controller.repeat();
  }

  @override
  void didUpdateWidget(covariant GlitterBox old) {
    super.didUpdateWidget(old);
    _controller.duration = widget.emergency
        ? const Duration(milliseconds: 1500)
        : const Duration(milliseconds: 3400);
    if (widget.active && !_controller.isAnimating) {
      _controller.repeat();
    } else if (!widget.active && _controller.isAnimating) {
      _controller.stop();
      _controller.value = 0;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final base = widget.emergency
        ? const [Color(0xFF4C1010), Color(0xFF7F1D1D), Color(0xFF4C1010)]
        : const [Color(0xFF2A1E0B), Color(0xFF1B1622), Color(0xFF2A1E0B)];
    final border = widget.emergency
        ? AppTheme.emergency
        : (widget.active ? AppTheme.warning : AppTheme.border);

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        final t = widget.active ? _controller.value : 0.0;
        return Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment(-1 + 2 * t, -0.4),
              end: Alignment(1 + 2 * t, 0.4),
              colors: widget.active
                  ? base
                  : const [Color(0xFF0F1626), Color(0xFF0F1626)],
            ),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: border),
            boxShadow: widget.active
                ? [
                    BoxShadow(
                      color: border.withAlpha(widget.emergency ? 90 : 45),
                      blurRadius: 18 * (0.6 + 0.4 * math.sin(t * math.pi)),
                    ),
                  ]
                : null,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                widget.title,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.6,
                  color: widget.emergency
                      ? AppTheme.emergency
                      : (widget.active ? AppTheme.warning : AppTheme.textMuted),
                ),
              ),
              const SizedBox(height: 6),
              Text(widget.body, style: const TextStyle(fontSize: 11.5)),
              if (widget.meta.isNotEmpty) ...[
                const SizedBox(height: 6),
                Text(
                  widget.meta,
                  style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
                ),
              ],
            ],
          ),
        );
      },
    );
  }
}

/// Inline emergency notice with a dismiss action. Deliberately *not* modal.
class _EmergencyChip extends StatelessWidget {
  const _EmergencyChip({required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final headline = state.emergency?['headline']?.toString() ??
        state.signal?.emergencyHeadline ??
        '';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF7F1D1D), Color(0xFFB45309)],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.emergency),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('\u{26A1}', style: TextStyle(fontSize: 12)),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              headline.isEmpty
                  ? 'emergency override — exit the position'
                  : 'emergency exit: “$headline”',
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600),
            ),
          ),
          const SizedBox(width: 6),
          GestureDetector(
            onTap: state.clearEmergency,
            child: const Text(
              'dismiss',
              style: TextStyle(
                fontSize: 10.5,
                decoration: TextDecoration.underline,
                color: AppTheme.textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Kv {
  const _Kv(this.label, this.value, {this.color, this.wide = false, this.small = false});

  final String label;
  final String value;
  final Color? color;
  final bool wide;
  final bool small;
}

class _KvGrid extends StatelessWidget {
  const _KvGrid({required this.items});

  final List<_Kv> items;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      runSpacing: 8,
      spacing: 12,
      children: items.map((item) {
        return SizedBox(
          width: item.wide ? double.infinity : 148,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                item.label.toUpperCase(),
                style: const TextStyle(
                  color: AppTheme.textMuted,
                  fontSize: 9.5,
                  letterSpacing: 0.8,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                item.value,
                style: TextStyle(
                  color: item.color ?? AppTheme.textPrimary,
                  fontSize: item.small ? 11 : 14.5,
                  fontWeight: FontWeight.w600,
                  fontFeatures:
                      item.small ? null : const [FontFeature.tabularFigures()],
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}

/// Countdown ring: the fraction is the *server* window's remaining time.
class _RingPainter extends CustomPainter {
  const _RingPainter({
    required this.fraction,
    required this.color,
    required this.urgent,
  });

  final double fraction;
  final Color color;
  final bool urgent;

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = (size.shortestSide - 8) / 2;
    final track = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 7
      ..color = AppTheme.border;
    final arc = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 7
      ..strokeCap = StrokeCap.round
      ..color = urgent ? AppTheme.warning : color;

    canvas.drawCircle(center, radius, track);
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2,
      2 * math.pi * fraction.clamp(0.0, 1.0),
      false,
      arc,
    );
  }

  @override
  bool shouldRepaint(covariant _RingPainter old) =>
      old.fraction != fraction || old.color != color || old.urgent != urgent;
}


/// The freshness chip: how old the prediction is against the 15-second rule.
class _FreshnessChip extends StatelessWidget {
  const _FreshnessChip({required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final prediction = state.prediction;
    if (prediction.ageSeconds == null) {
      return const SizedBox.shrink();
    }
    final stale = state.predictionStale;
    final label = stale
        ? 'STALE ${state.predictionAgeSeconds.toStringAsFixed(0)}s'
        : 'updated ${state.predictionAgeSeconds.toStringAsFixed(0)}s ago';
    final color = stale ? AppTheme.sell : AppTheme.buy;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
      decoration: BoxDecoration(
        border: Border.all(color: color.withAlpha(140)),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        label,
        style: TextStyle(color: color, fontSize: 9.5, letterSpacing: 0.4),
      ),
    );
  }
}

/// The reasoning bullets: what supports the side, then what argues against it.
class _ReasonList extends StatelessWidget {
  const _ReasonList({required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final bullets = state.prediction.reasoning.bullets;
    if (bullets.isEmpty) {
      return const Text(
        'reasoning unavailable for this window',
        style: TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
      );
    }
    final ordered = [...bullets]..sort(
        (a, b) => (b.supports ? 1 : 0).compareTo(a.supports ? 1 : 0),
      );
    return ConstrainedBox(
      constraints: const BoxConstraints(maxHeight: 92),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (final bullet in ordered.take(6))
              Padding(
                padding: const EdgeInsets.only(bottom: 3),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      bullet.supports ? '\u{25B8} ' : '\u{25BE} ',
                      style: TextStyle(
                        color: bullet.supports ? AppTheme.accent : AppTheme.sell,
                        fontSize: 11,
                      ),
                    ),
                    Expanded(
                      child: Text(
                        bullet.text,
                        style: const TextStyle(
                          color: AppTheme.textMuted,
                          fontSize: 10.5,
                          height: 1.35,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
