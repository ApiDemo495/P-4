import 'package:flutter/material.dart';

import '../models/signal.dart';
import '../state/app_state.dart';
import '../theme.dart';

/// The FROZEN signal panel.
///
/// While the cycle is COMPUTING it shows a spinner and the ⏳ icon: the client
/// has no signal to show, and the backend refuses to serve one. Once locked,
/// nothing on this panel may change until the next `CYCLE_START`.
class SignalPanel extends StatelessWidget {
  const SignalPanel({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final signal = state.signal;
    if (state.isComputing || signal == null) {
      return const Panel(
        child: SizedBox(
          height: 190,
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 26,
                  height: 26,
                  child: CircularProgressIndicator(strokeWidth: 2.4),
                ),
                SizedBox(height: 14),
                Text('\u{23F3} Preparing the first window…',
                    style: TextStyle(color: AppTheme.textMuted)),
                SizedBox(height: 6),
                Text(
                    'the engine computes each signal during the previous countdown, '
                    'so once it locks the panel never blanks',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: AppTheme.textMuted, fontSize: 11)),
              ],
            ),
          ),
        ),
      );
    }

    final color = AppTheme.forSignal(signal.signal);
    return Panel(
      title: 'Signal · ${signal.asset} · locked at ${signal.timestamp}',
      trailing: Text(signal.lockIcon, style: const TextStyle(fontSize: 18)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                decoration: BoxDecoration(
                  color: color.withAlpha(38),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: color),
                ),
                child: Text(
                  signal.signal,
                  style: TextStyle(
                    color: color,
                    fontSize: 26,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1.5,
                  ),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'confidence ${(signal.confidence * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 6),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(5),
                      child: LinearProgressIndicator(
                        value: signal.confidence.clamp(0.0, 1.0),
                        minHeight: 8,
                        backgroundColor: AppTheme.background,
                        valueColor: AlwaysStoppedAnimation<Color>(color),
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      '\$${signal.price.toStringAsFixed(2)} · formulas '
                      '${signal.totalMs.toStringAsFixed(1)} ms · DRG '
                      '${signal.drg.toStringAsFixed(3)}',
                      style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (signal.isEmergencyOverride) ...[
            const SizedBox(height: 14),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppTheme.emergency.withAlpha(28),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppTheme.emergency),
              ),
              child: Text(
                '\u{26A1} EMERGENCY OVERRIDE → HOLD\n${signal.emergencyHeadline}\n'
                'Superseded: ${signal.supersededBy ?? 'COMPUTING'}',
                style: const TextStyle(color: AppTheme.emergency, fontSize: 12.5),
              ),
            ),
          ],
          const SizedBox(height: 14),
          Text(signal.reasoning,
              style: const TextStyle(color: AppTheme.textPrimary, fontSize: 12.5)),
          const SizedBox(height: 12),
          Text(
            'BRAIN (full wiring: Brain tab · /api/brain/wiring)',
            style: const TextStyle(
              fontSize: 10.5,
              letterSpacing: 1.2,
              color: AppTheme.textMuted,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          StatRow(label: 'CCSv2', value: signal.ccsValue.toStringAsFixed(3)),
          StatRow(
            label: 'confidence (KC sparsity × read-out margin)',
            value: (signal.ccsConfidence * 100).toStringAsFixed(1) + '%',
          ),
          StatRow(label: 'brain status', value: signal.brainStatus),
          const SizedBox(height: 10),
          if (signal.weightsUsed.isNotEmpty)
            Text(
              signal.weightsUsed.entries
                  .map((e) => '${e.key} ${(e.value * 100).toStringAsFixed(0)}%')
                  .join(' · '),
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
            ),
          if (signal.warnings.isNotEmpty) ...[
            const SizedBox(height: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: signal.warnings
                  .take(3)
                  .map((w) => Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('⚠ ', style: TextStyle(color: AppTheme.warning)),
                          Expanded(
                            child: Text(w,
                                style: const TextStyle(
                                    color: AppTheme.warning, fontSize: 11.5)),
                          ),
                        ],
                      ))
                  .toList(),
            ),
          ],
          if (signal.holdWarning != null) ...[
            const SizedBox(height: 12),
            HoldWarningBox(warning: signal.holdWarning!),
          ],
        ],
      ),
    );
  }
}

/// The verbatim Section 10.2 HOLD box.
class HoldWarningBox extends StatelessWidget {
  const HoldWarningBox({super.key, required this.warning});

  final HoldWarning warning;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.warning.withAlpha(22),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.warning.withAlpha(120)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('⚠  HOLD signal',
              style: TextStyle(
                  color: AppTheme.warning, fontWeight: FontWeight.w700, fontSize: 12.5)),
          const SizedBox(height: 6),
          Text(warning.text,
              style: const TextStyle(color: AppTheme.textPrimary, fontSize: 12)),
          if (warning.lean != null) ...[
            const SizedBox(height: 8),
            Text(
              'Lean direction: ${warning.lean} '
              '(score ${warning.leanScore.toStringAsFixed(3)}, '
              'confidence ${(warning.confidence * 100).toStringAsFixed(0)}%)',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
            ),
          ],
        ],
      ),
    );
  }
}
