import 'package:flutter/material.dart';

import '../models/signal.dart';
import '../state/app_state.dart';
import '../theme.dart';

/// The four fusion inputs with their weights, status and this cycle's call.
class AgentList extends StatelessWidget {
  const AgentList({super.key, required this.state});

  final AppState state;

  static const _labels = {
    'drosophila': '\u{1F9E0}  Drosophila CCSv2',
    'gemini': '\u{1F916}  Gemini AI',
    'local': '\u{1F4BB}  Local model',
    'github': '\u{1F419}  GitHub Models',
  };

  @override
  Widget build(BuildContext context) {
    final signal = state.signal;
    final rows = <Widget>[];

    for (final entry in _labels.entries) {
      final name = entry.key;
      final weight = state.weights[name] ?? 0.0;
      AgentDecision? decision;
      if (name == 'drosophila') {
        if (signal != null) {
          decision = AgentDecision(
            name: name,
            decision: signal.signal,
            confidence: signal.ccsConfidence,
            status: 'LIVE',
            model: '80-node mushroom body',
          );
        }
      } else {
        decision = signal?.agents[name];
      }
      final liveStatus = decision?.status ??
          (state.agentStatus[name]?['status']?.toString() ?? 'DISABLED');

      rows.add(
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(entry.value,
                        style: const TextStyle(fontWeight: FontWeight.w600)),
                    const SizedBox(height: 2),
                    Text(
                      decision?.model.isNotEmpty == true
                          ? decision!.model
                          : (state.agentStatus[name]?['detail']?.toString() ?? ''),
                      style: const TextStyle(
                          color: AppTheme.textMuted, fontSize: 11),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              _statusChip(liveStatus),
              const SizedBox(width: 10),
              SizedBox(
                width: 108,
                child: Text(
                  decision?.decision == null
                      ? (decision?.error.isNotEmpty == true
                          ? decision!.error.substring(
                              0,
                              decision.error.length > 22 ? 22 : decision.error.length)
                          : '—')
                      : '${decision!.decision} '
                          '${((decision.confidence ?? 0) * 100).toStringAsFixed(0)}%',
                  textAlign: TextAlign.right,
                  style: const TextStyle(fontSize: 11.5),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 42,
                child: Text(
                  '${(weight * 100).toStringAsFixed(0)}%',
                  textAlign: TextAlign.right,
                  style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Panel(title: 'Agents & fusion weights', child: Column(children: rows));
  }

  Widget _statusChip(String status) {
    late Color color;
    switch (status) {
      case 'ACTIVE':
      case 'LIVE':
        color = AppTheme.buy;
        break;
      case 'STUB':
        color = AppTheme.warning;
        break;
      case 'ERROR':
        color = AppTheme.sell;
        break;
      case 'TIMEOUT':
      case 'RATE_LIMITED':
      case 'INACTIVE':
        color = AppTheme.emergency;
        break;
      default:
        color = AppTheme.textMuted;
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
      decoration: BoxDecoration(
        color: color.withAlpha(30),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withAlpha(120)),
      ),
      child: Text(status,
          style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w700)),
    );
  }
}

/// Full-screen red/amber overlay shown while an emergency override is active.
class EmergencyOverlay extends StatelessWidget {
  const EmergencyOverlay({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final event = state.emergency;
    if (event == null) return const SizedBox.shrink();
    return Positioned.fill(
      child: Container(
        color: AppTheme.emergency.withAlpha(28),
        child: Center(
          child: Container(
            width: 520,
            margin: const EdgeInsets.all(24),
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(
              color: AppTheme.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppTheme.emergency, width: 2),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('\u{26A1}  EMERGENCY OVERRIDE',
                    style: TextStyle(
                        color: AppTheme.emergency,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2)),
                const SizedBox(height: 12),
                Text(event['headline']?.toString() ?? '',
                    style: const TextStyle(fontSize: 15, height: 1.35)),
                const SizedBox(height: 8),
                Text(
                  'Previous signal: ${event['previous_signal']} → HOLD · '
                  '${state.emergencyRemaining.toStringAsFixed(0)}s remaining',
                  style: const TextStyle(color: AppTheme.textMuted, fontSize: 12),
                ),
                const SizedBox(height: 16),
                Align(
                  alignment: Alignment.centerRight,
                  child: FilledButton.tonal(
                    onPressed: state.clearEmergency,
                    child: const Text('Clear override'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
