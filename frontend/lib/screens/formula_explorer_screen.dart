import 'package:flutter/material.dart';

import '../models/signal.dart';
import '../state/app_state.dart';
import '../theme.dart';

/// All 22 formulas, grouped in the 8 categories, with live values.
///
/// The banner makes the Rule-2 contract explicit: these numbers refresh every
/// 15 seconds, the locked signal above them does not.
class FormulaExplorerScreen extends StatefulWidget {
  const FormulaExplorerScreen({super.key, required this.state});

  final AppState state;

  @override
  State<FormulaExplorerScreen> createState() => _FormulaExplorerScreenState();
}

class _FormulaExplorerScreenState extends State<FormulaExplorerScreen> {
  bool _showDescriptions = false;
  final Set<String> _collapsed = {};

  @override
  Widget build(BuildContext context) {
    final state = widget.state;
    final values = state.liveFormulas.isNotEmpty
        ? state.liveFormulas
        : state.lastLockedFormulas;
    final totalMs = (state.timings['total_ms'] as num?)?.toDouble();

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        Panel(
          child: Row(
            children: [
              const Icon(Icons.lock_outline, size: 16, color: AppTheme.textMuted),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Live formula values only — the signal remains '
                  '${state.lockState} (${state.lockIcon}) until the next cycle.'
                  '${totalMs != null ? '  Total pass ${totalMs.toStringAsFixed(2)} ms.' : ''}',
                  style: const TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
                ),
              ),
              TextButton(
                onPressed: () =>
                    setState(() => _showDescriptions = !_showDescriptions),
                child: Text(_showDescriptions ? 'Hide descriptions' : 'Descriptions'),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        ...state.catalog.map((category) => _category(category, values)),
      ],
    );
  }

  Widget _category(FormulaCategory category, Map<String, double> values) {
    final collapsed = _collapsed.contains(category.key);
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Panel(
        padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            InkWell(
              onTap: () => setState(() {
                if (collapsed) {
                  _collapsed.remove(category.key);
                } else {
                  _collapsed.add(category.key);
                }
              }),
              child: Row(
                children: [
                  Icon(collapsed ? Icons.chevron_right : Icons.expand_more,
                      size: 18, color: AppTheme.textMuted),
                  const SizedBox(width: 4),
                  Expanded(
                    child: Text(
                      'Category ${category.key} — ${category.name}',
                      style: const TextStyle(
                          fontWeight: FontWeight.w700, fontSize: 13),
                    ),
                  ),
                  Text('${category.formulas.length} formulas',
                      style: const TextStyle(
                          color: AppTheme.textMuted, fontSize: 11)),
                ],
              ),
            ),
            if (!collapsed) const SizedBox(height: 10),
            if (!collapsed)
              ...category.formulas
                  .map((spec) => _row(spec, values[spec.name] ?? 0.0)),
          ],
        ),
      ),
    );
  }

  Widget _row(FormulaSpec spec, double value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              SizedBox(
                width: 62,
                child: Text(spec.name,
                    style: const TextStyle(
                        fontWeight: FontWeight.w700, fontSize: 12.5)),
              ),
              Expanded(child: SignedBar(value: value, min: -1, max: 1)),
              const SizedBox(width: 10),
              SizedBox(
                width: 58,
                child: Text(
                  value.toStringAsFixed(3),
                  textAlign: TextAlign.right,
                  style: TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: spec.directional ? AppTheme.forValue(value) : AppTheme.accent,
                  ),
                ),
              ),
            ],
          ),
          if (_showDescriptions) ...[
            const SizedBox(height: 4),
            Padding(
              padding: const EdgeInsets.only(left: 62),
              child: Text(
                '${spec.title} · brain node: ${spec.brainNode}\n'
                '${spec.description}  (budget ${spec.budgetMs} ms, '
                '${spec.directional ? 'directional' : 'regime/confidence'})',
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
