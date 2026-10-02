import 'package:flutter/material.dart';

import '../theme.dart';

/// The thermodynamic capital layer (Round T) - BTC as consumed work, PAXG as
/// inert rest mass - exactly as it was locked with the window on screen.
///
/// Fed by `physics` inside every snapshot / pulse (and `GET
/// /api/physics/current`).  It is one weighted voter in the fusion beside the
/// brain and the AI agents, never a veto; every reading carries its source
/// (live telemetry, the frozen tape, or a model constant).
class PhysicsPanel extends StatelessWidget {
  const PhysicsPanel({super.key, required this.payload});

  final Map<String, dynamic> payload;

  @override
  Widget build(BuildContext context) {
    final report = payload['report'];
    if (report is! Map) {
      return Panel(
        title: 'Thermodynamic capital layer',
        child: Text(
          payload['weight'] == 0
              ? 'disabled (PHYSICS_WEIGHT=0)'
              : 'waiting for the first locked window…',
          style: const TextStyle(color: AppTheme.textMuted, fontSize: 12),
        ),
      );
    }
    final r = Map<String, dynamic>.from(report);
    final w = _map(r['weights']);
    final comp = _map(r['composite']);
    final physical = _map(r['physical']);
    final landauer = _map(physical['landauer']);
    final solar = _map(physical['solar']);
    final energy = _map(physical['energy_mass']);
    final vote = _d(r['vote']);
    final side = vote >= 0 ? 'BUY' : 'SELL';
    final mechanisms = (r['mechanisms'] as List?) ?? const [];
    final locked = payload['locked'] == true;

    return Panel(
      title: 'Thermodynamic capital layer',
      trailing: Text(
        '${locked ? '🔒 locked' : 'live'} · weight ${_pct(_d(payload['weight']))}',
        style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                '$side ${r['asset']}',
                style: TextStyle(
                  color: side == 'BUY' ? AppTheme.buy : AppTheme.sell,
                  fontWeight: FontWeight.w800,
                  fontSize: 16,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                'vote ${vote.toStringAsFixed(3)} at ${_pct(_d(r['confidence']))}',
                style: const TextStyle(fontSize: 12),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _bar(_d(w['w_composite']), _d(w['w_micro'])),
          const SizedBox(height: 4),
          Text(
            'target BTC weight ${_pct(_d(w['w_composite']))} '
            '(thermal ${_pct(_d(w['w_thermal']))} · solar ${_pct(_d(w['w_solar']))} · '
            'α ${_d(w['alpha']).toStringAsFixed(2)}) · microstructure ${_pct(_d(w['w_micro']))}'
            '${w['clamped'] == true ? ' · clamped' : ''}',
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
          ),
          const SizedBox(height: 10),
          StatRow(
              label: 'Landauer Θ',
              value: '${_d(landauer['theta']).toStringAsFixed(4)} (Θ* ${landauer['theta_star']})'),
          StatRow(
              label: 'Solar Ω',
              value: '${_d(solar['omega']).toStringAsFixed(4)} · '
                  '${(_d(solar['sunlit_share']) * 100).round()}% of hashrate in sunlight'),
          StatRow(
              label: 'E = mc² ratio',
              value: '${_d(energy['ratio_oz_per_btc']).toStringAsFixed(1)} oz-eq/BTC '
                  'vs market ${_d(energy['market_ratio']).toStringAsFixed(2)}'),
          StatRow(
              label: 'Phase Φ',
              value: '${_d(comp['phase_angle_deg']).toStringAsFixed(2)}° — ${comp['phase'] ?? ''}'),
          StatRow(
              label: 'TSR · edge',
              value: '${_d(comp['tsr']).toStringAsFixed(4)} · '
                  '${_d(comp['expected_edge_bps']).toStringAsFixed(2)} bp expected'),
          StatRow(label: 'Live inputs', value: '${r['live_inputs'] ?? 0} telemetry sources'),
          const SizedBox(height: 10),
          ...mechanisms.map((m) {
            final row = Map<String, dynamic>.from(m as Map);
            final dir = (row['direction'] as num?)?.toInt() ?? 0;
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 3),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    width: 30,
                    child: Text('§${row['section']}',
                        style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('${row['name']}',
                            style: const TextStyle(fontSize: 11.5, fontWeight: FontWeight.w600)),
                        Text('${row['logic'] ?? ''}',
                            style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
                      ],
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    dir > 0 ? 'BTC +' : dir < 0 ? 'PAXG +' : '—',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: dir > 0
                          ? AppTheme.buy
                          : dir < 0
                              ? AppTheme.sell
                              : AppTheme.textMuted,
                    ),
                  ),
                ],
              ),
            );
          }),
          const SizedBox(height: 8),
          Text('${landauer['logic'] ?? ''}',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
          Text('${solar['logic'] ?? ''}',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
          Text('${energy['logic'] ?? ''}',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
          Text('${w['logic'] ?? ''}',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5)),
          const SizedBox(height: 6),
          Text('${comp['note'] ?? ''}',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 10)),
        ],
      ),
    );
  }

  Widget _bar(double physicsWeight, double microWeight) {
    return LayoutBuilder(builder: (context, constraints) {
      final width = constraints.maxWidth;
      return SizedBox(
        height: 14,
        child: Stack(
          children: [
            Container(
              height: 10,
              margin: const EdgeInsets.only(top: 2),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(6),
                gradient: const LinearGradient(
                    colors: [Color(0xFFD4A017), Color(0xFF1F2A3A), Color(0xFFF7931A)]),
              ),
            ),
            Positioned(
              left: (width * physicsWeight.clamp(0.0, 1.0)) - 1,
              child: Container(width: 2, height: 14, color: Colors.white),
            ),
            Positioned(
              left: (width * microWeight.clamp(0.0, 1.0)) - 2,
              child: Container(width: 4, height: 14, color: AppTheme.buy),
            ),
          ],
        ),
      );
    });
  }

  static Map<String, dynamic> _map(dynamic v) =>
      v is Map ? Map<String, dynamic>.from(v) : const {};

  static double _d(dynamic v) => v is num ? v.toDouble() : 0.0;

  static String _pct(double v) => '${(v * 100).toStringAsFixed(0)}%';
}
