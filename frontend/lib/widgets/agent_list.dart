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

// NOTE: the emergency notice used to be a full-screen overlay here.  It is now
// an inline, glittering conviction box inside the fixed-layout widget panel
// (`signal_widget_panel.dart`), so nothing ever covers the dashboard.
