import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';

/// Hedge dashboard: the four BTC×PAXG formulas plus the brain read-out.
class HedgeDashboard extends StatelessWidget {
  const HedgeDashboard({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final values = state.liveFormulas.isNotEmpty
        ? state.liveFormulas
        : state.lastLockedFormulas;
    double value(String key) => values[key] ?? 0.0;

    final hsi = value('HSI');
    final stressed = hsi > 0.8;
    final signal = state.signal;

    return Panel(
      title: 'Hedge · BTC × PAXG',
      trailing: Text(
        stressed ? 'HIGH STRESS' : 'low stress',
        style: TextStyle(
          fontSize: 10.5,
          fontWeight: FontWeight.w700,
          color: stressed ? AppTheme.sell : AppTheme.buy,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _metric('HSI · Hedge Stress Index', hsi, stressed: stressed),
          _metric('HRDD · Hedge Ratio Drift', value('HRDD')),
          _metric('SHRP · Safe-Haven Rotation', value('SHRP')),
          _metric('GCDV · Divergence Velocity', value('GCDV')),
          const SizedBox(height: 6),
          const Divider(color: AppTheme.border, height: 18),
          StatRow(
            label: 'CCSv2 (locked)',
            value: (signal?.ccsValue ?? value('CCSv2')).toStringAsFixed(3),
            color: AppTheme.forValue(signal?.ccsValue ?? value('CCSv2')),
          ),
          StatRow(
            label: 'brain status',
            value: state.brainStatus['status']?.toString() ??
                signal?.brainStatus ??
                '—',
          ),
          StatRow(
            label: 'matrix checksum',
            value: (state.brainStatus['matrix']?['checksum'] ?? '—').toString(),
          ),
          StatRow(label: 'DRG reward', value: value('DRG').toStringAsFixed(3)),
          StatRow(
            label: 'win rate',
            value:
                '${(state.winRate * 100).toStringAsFixed(0)}% (${state.outcomeCount} outcomes)',
          ),
        ],
      ),
    );
  }

  Widget _metric(String label, double value, {bool stressed = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5)),
              Text(
                value.toStringAsFixed(3),
                style: TextStyle(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w700,
                  color: stressed ? AppTheme.sell : AppTheme.forValue(value),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          SignedBar(value: value, min: -1, max: 1),
        ],
      ),
    );
  }
}
