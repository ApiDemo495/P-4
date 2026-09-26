import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';

/// "Where is the fly brain actually used?" - the answer, in the app.
///
/// The screen is fed by two endpoints that exist for exactly this purpose:
///
///   GET /api/brain/wiring   the static map: which formula drives which neuron,
///                           the five circuit stages and the fusion weight;
///   GET /api/brain/explain  what the circuit did in the window that is locked
///                           *right now* - dominant inputs, Kenyon-cell
///                           sparsity, MBON/LH read-outs, the dopamine gates and
///                           the share of the final score the brain owns.
///
/// Everything is live data from the running engine, so the wiring claim
/// ("40 % of the decision rides on the mushroom body") is verifiable rather
/// than decorative.
class BrainScreen extends StatelessWidget {
  const BrainScreen({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final wiring = state.wiring;
    final explain = state.brainExplain;
    final available = explain['available'] == true;

    return RefreshIndicator(
      onRefresh: () async {
        await state.refreshWiring();
        await state.refreshBrainExplain();
        await state.refreshBrain();
      },
      child: ListView(
        padding: const EdgeInsets.all(14),
        children: [
          Panel(
            title: 'Drosophila circuit · how it is wired',
            trailing: Text(
              'fusion weight ${wiring['fusion_weight'] ?? 0.4}',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
            ),
            child: Column(
              children: _stages(wiring, explain, available),
            ),
          ),
          const SizedBox(height: 12),
          Panel(
            title: 'This window, neuron by neuron',
            trailing: Text(
              available ? 'cycle ${explain['cycle_number'] ?? '—'}' : 'waiting',
              style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
            ),
            child: available ? _live(explain) : _waiting(explain),
          ),
          const SizedBox(height: 12),
          _wiringMap(wiring),
        ],
      ),
    );
  }

  // ------------------------------------------------------------------ stages
  List<Widget> _stages(
    Map<String, dynamic> wiring,
    Map<String, dynamic> explain,
    bool available,
  ) {
    final stages = (wiring['stages'] as List?) ?? const [];
    if (stages.isEmpty) {
      return const [
        Text('loading the wiring map…',
            style: TextStyle(color: AppTheme.textMuted, fontSize: 12)),
      ];
    }
    final live = <String>[
      available
          ? '${((explain['all_inputs'] as Map?) ?? const {}).length}/20 formulas '
              'written onto the projection neurons'
          : '20 formulas → PNs 0–19',
      available
          ? '${explain['kenyon_cells']?['active'] ?? 0}/'
              '${explain['kenyon_cells']?['of'] ?? 50} KC clusters active (top 10 %)'
          : '50 KC clusters · ReLU + top-10 % sparsity',
      available
          ? 'DRG ${_s(explain['dopamine']?['drg'])} → PAM · '
              'OA ${_s(explain['dopamine']?['octopamine'])} (HSI gated)'
          : 'DRG → PAM / PPL1 · HSI → OA',
      available
          ? 'approach ${_s(explain['mbons']?['approach'])} · '
              'avoid ${_s(explain['mbons']?['avoid'])} · '
              'confidence ${_pct(explain['mbons']?['confidence'])}'
          : '4 MBONs · 3 graph-convolution layers · gain 3.2802',
      available
          ? 'CCSv2 ${_s(explain['ccs_value'])} @ ${_pct(explain['ccs_confidence'])} '
              '→ ${wiring['fusion_weight'] ?? 0.4} of the fused decision'
          : 'CCSv2 + KCAE → fusion (0.40 weight)',
    ];

    final out = <Widget>[];
    for (var i = 0; i < stages.length; i++) {
      final stage = Map<String, dynamic>.from(stages[i] as Map);
      out.add(
        Container(
          margin: const EdgeInsets.only(bottom: 8),
          padding: const EdgeInsets.all(11),
          decoration: BoxDecoration(
            color: AppTheme.background,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: AppTheme.border),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 26,
                padding: const EdgeInsets.symmetric(vertical: 3),
                decoration: BoxDecoration(
                  color: AppTheme.surfaceAlt,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: AppTheme.border),
                ),
                child: Text(
                  '${i + 1}',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: AppTheme.accent, fontSize: 11, fontWeight: FontWeight.w700),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      (stage['name'] ?? '').toString(),
                      style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12.5),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      (stage['role'] ?? '').toString(),
                      style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      (stage['detail'] ?? '').toString(),
                      style: const TextStyle(color: Color(0xFF5C6B86), fontSize: 10.5),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '▸ ${live[i]}',
                      style: const TextStyle(color: AppTheme.accent, fontSize: 10.5),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      );
    }
    return out;
  }

  // -------------------------------------------------------------------- live
  Widget _live(Map<String, dynamic> explain) {
    final dominant = ((explain['dominant_inputs'] as List?) ?? const [])
        .take(5)
        .map((row) {
      final r = Map<String, dynamic>.from(row as Map);
      return '${r['formula']} ${_s(r['value'])}';
    }).join(' · ');

    final weights = (explain['weights'] as Map?) ?? const {};
    final source = (explain['source'] as Map?) ?? const {};

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(11),
          decoration: BoxDecoration(
            color: AppTheme.surfaceAlt,
            borderRadius: BorderRadius.circular(10),
            border: const Border(
              left: BorderSide(color: AppTheme.accent, width: 3),
            ),
          ),
          child: Text(
            (explain['verdict'] ?? '').toString(),
            style: const TextStyle(fontSize: 12.5, height: 1.35),
          ),
        ),
        const SizedBox(height: 10),
        StatRow(label: 'Dominant inputs', value: dominant.isEmpty ? '—' : dominant),
        StatRow(
          label: 'Kenyon cells active',
          value: '${explain['kenyon_cells']?['active'] ?? '—'} / '
              '${explain['kenyon_cells']?['of'] ?? 50} '
              '(KCAE ${_s(explain['kenyon_cells']?['kcae'])})',
        ),
        StatRow(
          label: 'MBON approach / avoid',
          value: '${_s(explain['mbons']?['approach'])} / '
              '${_s(explain['mbons']?['avoid'])}',
        ),
        StatRow(
          label: 'Lateral horn',
          value: 'approach ${_s(explain['lateral_horn']?['approach'])} · '
              'avoid ${_s(explain['lateral_horn']?['avoid'])} · '
              'neutral ${_s(explain['lateral_horn']?['neutral'])}',
        ),
        StatRow(
          label: 'Dopamine / octopamine gates',
          value: 'PAM ${_s(explain['dopamine']?['pam'])} · '
              'PPL1 ${_s(explain['dopamine']?['ppl1'])} · '
              'OA ${_s(explain['dopamine']?['octopamine'])}',
        ),
        StatRow(
          label: 'CCSv2 → fusion',
          value: '${_s(explain['ccs_value'])} × '
              '${weights['in_fusion'] ?? 0.4} = '
              '${_s(weights['share_of_score'])} of the score '
              '(${weights['decision'] ?? '—'})',
        ),
        const SizedBox(height: 8),
        Text(
          'connectome: ${source['status'] ?? '—'} · ${source['message'] ?? ''} · '
          'checksum ${source['checksum'] ?? '—'} · gain ${source['gain'] ?? '—'}',
          style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
        ),
        if ((source['steps'] as List?)?.isNotEmpty ?? false) ...[
          const SizedBox(height: 8),
          ...(source['steps'] as List).map((step) {
            final s = Map<String, dynamic>.from(step as Map);
            final ok = s['ok'] == true;
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                children: [
                  Text(ok ? '✔' : '✘',
                      style: TextStyle(
                          color: ok ? AppTheme.buy : AppTheme.textMuted, fontSize: 11)),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      '${s['step']} · ${s['detail']} '
                      '(${((s['elapsed_ms'] as num?) ?? 0).toStringAsFixed(1)} ms)',
                      style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ],
    );
  }

  Widget _waiting(Map<String, dynamic> explain) {
    return Text(
      (explain['detail'] ??
              'the first window is still being computed — the trace appears here '
                  'the moment it locks')
          .toString(),
      style: const TextStyle(color: AppTheme.textMuted, fontSize: 12),
    );
  }

  // ------------------------------------------------------------- wiring map
  Widget _wiringMap(Map<String, dynamic> wiring) {
    final rows = (wiring['projection_neurons'] as List?) ?? const [];
    final layout = (wiring['layout'] as Map?) ?? const {};
    return Panel(
      title: 'Formula → neuron map',
      trailing: Text(
        '${rows.length} projection neurons',
        style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (rows.isEmpty)
            const Text('loading…',
                style: TextStyle(color: AppTheme.textMuted, fontSize: 12)),
          ...rows.map((row) {
            final r = Map<String, dynamic>.from(row as Map);
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 3),
              child: Row(
                children: [
                  SizedBox(
                    width: 54,
                    child: Text(
                      'PN ${r['pn']}',
                      style: const TextStyle(
                          color: AppTheme.accent, fontSize: 11, fontWeight: FontWeight.w700),
                    ),
                  ),
                  SizedBox(
                    width: 62,
                    child: Text(
                      (r['formula'] ?? '').toString(),
                      style: const TextStyle(fontSize: 11.5, fontWeight: FontWeight.w600),
                    ),
                  ),
                  Expanded(
                    child: Text(
                      (r['brain_node'] ?? r['title'] ?? '').toString(),
                      style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
                    ),
                  ),
                ],
              ),
            );
          }),
          const SizedBox(height: 10),
          Text(
            'layout: PNs ${_range(layout['pn'])} · KC ${_range(layout['kenyon_cells'])} · '
            'DANs ${_range(layout['dans'])} · MBONs ${_range(layout['mbons'])} · '
            'lateral horn ${_range(layout['lateral_horn'])} · '
            'KC sparsity ${wiring['sparsity'] ?? '—'}',
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 10.5),
          ),
        ],
      ),
    );
  }

  static String _range(dynamic value) {
    if (value is List && value.length == 2) return '${value[0]}–${value[1]}';
    return '—';
  }

  static String _s(dynamic value) {
    if (value == null) return '—';
    if (value is num) {
      final v = value.toDouble();
      return '${v >= 0 ? '+' : ''}${v.toStringAsFixed(3)}';
    }
    return value.toString();
  }

  static String _pct(dynamic value) {
    if (value is num) return '${(value.toDouble() * 100).toStringAsFixed(0)}%';
    final parsed = double.tryParse('$value');
    return parsed == null ? '—' : '${(parsed * 100).toStringAsFixed(0)}%';
  }
}
