import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';

/// The cycle countdown with the lock icon of the Signal Lock Protocol.
///
/// ⏳ COMPUTING · 🔒 LOCKED · ⚡ EMERGENCY_OVERRIDE
class CycleTimer extends StatelessWidget {
  const CycleTimer({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    // A 60-second window: the number is the seconds left to the boundary, on
    // the one clock the whole engine ticks to (the backend publishes the window
    // as absolute instants, so this cannot drift or restart).
    final total = state.cyclePeriodSeconds <= 0 ? 60.0 : state.cyclePeriodSeconds;
    final elapsed = (total - state.secondsRemaining).clamp(0.0, total);
    final remaining = state.secondsRemaining;
    final minutes = (remaining ~/ 60).toString().padLeft(2, '0');
    final seconds = (remaining % 60).floor().toString().padLeft(2, '0');
    final color = state.isEmergency ? AppTheme.emergency : AppTheme.accent;

    return Panel(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '$minutes:$seconds',
                style: TextStyle(
                  fontSize: 34,
                  fontWeight: FontWeight.w700,
                  color: color,
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
              Text(state.lockIcon, style: const TextStyle(fontSize: 28)),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: (elapsed / total).clamp(0.0, 1.0),
              minHeight: 6,
              backgroundColor: AppTheme.background,
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
          const SizedBox(height: 10),
          Text(
            state.lockLabel,
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 12),
          ),
          const SizedBox(height: 4),
          Text(
            state.countdownNote,
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
          ),
          const SizedBox(height: 6),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('cycle ${state.cycleNumber}',
                  style: const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
              Text(
                state.ntpSynced ? 'UTC · NTP synced' : 'UTC · system clock',
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Icon(
                state.connected ? Icons.wifi_tethering : Icons.wifi_tethering_off,
                size: 13,
                color: state.connected ? AppTheme.buy : AppTheme.sell,
              ),
              const SizedBox(width: 5),
              Text(
                state.connected ? 'live' : 'reconnecting…',
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppTheme.surfaceAlt,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppTheme.border),
                ),
                child: Text(
                  'L${state.degradationLevel} ${state.degradationLabel}',
                  style: const TextStyle(color: AppTheme.warning, fontSize: 10.5),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
