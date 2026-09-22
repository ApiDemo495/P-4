import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../state/app_state.dart';
import '../theme.dart';
import '../widgets/agent_list.dart';
import '../widgets/cycle_timer.dart';
import '../widgets/hedge_dashboard.dart';
import '../widgets/news_card.dart';
import '../widgets/signal_panel.dart';
import 'formula_explorer_screen.dart';
import 'settings_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 14,
        title: Row(
          children: [
            const Text('\u{1FAB0}', style: TextStyle(fontSize: 18)),
            const SizedBox(width: 8),
            const Text('DROSOPHILA TRADER',
                style: TextStyle(fontSize: 15, letterSpacing: 1.1)),
            const SizedBox(width: 6),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
              decoration: BoxDecoration(
                color: AppTheme.surfaceAlt,
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Text('v2.0',
                  style: TextStyle(fontSize: 10, color: AppTheme.textMuted)),
            ),
          ],
        ),
        actions: [
          _assetToggle(state),
          const SizedBox(width: 8),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              if (state.pendingAsset != null && state.pendingAsset != state.asset)
                Container(
                  width: double.infinity,
                  color: AppTheme.warning.withAlpha(30),
                  padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 14),
                  child: Text(
                    'Switching to ${state.pendingAsset} at the next cycle boundary…',
                    style: const TextStyle(color: AppTheme.warning, fontSize: 12),
                  ),
                ),
              Expanded(
                child: IndexedStack(
                  index: _tab,
                  children: [
                    _signalTab(state),
                    FormulaExplorerScreen(state: state),
                    SettingsScreen(state: state),
                  ],
                ),
              ),
            ],
          ),
          EmergencyOverlay(state: state),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (index) => setState(() => _tab = index),
        backgroundColor: AppTheme.surface,
        destinations: const [
          NavigationDestination(
              icon: Icon(Icons.show_chart), label: 'Signal'),
          NavigationDestination(icon: Icon(Icons.functions), label: 'Formulas'),
          NavigationDestination(icon: Icon(Icons.settings), label: 'Settings'),
        ],
      ),
    );
  }

  /// BTC / PAXG toggle. The selection is persisted locally; the *engine*
  /// applies it at the next cycle boundary, never mid-cycle.
  Widget _assetToggle(AppState state) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        color: AppTheme.background,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: ['BTC', 'PAXG'].map((asset) {
          final selected = asset == state.asset;
          return GestureDetector(
            onTap: () => state.selectAsset(asset),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
              decoration: BoxDecoration(
                color: selected ? AppTheme.accent.withAlpha(40) : null,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '${selected ? '●' : '○'} $asset',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w400,
                  color: selected ? AppTheme.accent : AppTheme.textMuted,
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _signalTab(AppState state) {
    return RefreshIndicator(
      onRefresh: state.refreshPanels,
      child: ListView(
        padding: const EdgeInsets.all(14),
        children: [
          SignalPanel(state: state),
          const SizedBox(height: 12),
          LayoutBuilder(
            builder: (context, constraints) {
              final wide = constraints.maxWidth > 720;
              final left = Column(
                children: [
                  CycleTimer(state: state),
                  const SizedBox(height: 12),
                  HedgeDashboard(state: state),
                ],
              );
              final right = Column(
                children: [
                  AgentList(state: state),
                  const SizedBox(height: 12),
                  NewsCard(state: state),
                  const SizedBox(height: 12),
                  _history(state),
                  const SizedBox(height: 12),
                  _outcomes(state),
                ],
              );
              if (!wide) {
                return Column(
                  children: [left, const SizedBox(height: 12), right],
                );
              }
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(flex: 4, child: left),
                  const SizedBox(width: 12),
                  Expanded(flex: 6, child: right),
                ],
              );
            },
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _history(AppState state) {
    return Panel(
      title: 'Signal history',
      trailing: Text('${state.history.length} shown',
          style: const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
      child: Column(
        children: state.history.take(6).map((row) {
          final signal = (row['signal'] ?? 'HOLD').toString();
          final emergency = row['is_emergency_override'] == true;
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: Row(
              children: [
                SizedBox(
                  width: 96,
                  child: Text(
                    '${emergency ? '\u{26A1}' : '\u{1F512}'} ${emergency ? 'EMERGENCY' : signal}',
                    style: TextStyle(
                      fontSize: 11.5,
                      color: emergency
                          ? AppTheme.emergency
                          : AppTheme.forSignal(signal),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                Expanded(
                  child: Text(
                    '${row['asset']} · conf '
                    '${(((row['confidence'] as num?)?.toDouble() ?? 0) * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(fontSize: 11.5),
                  ),
                ),
                Text(
                  '#${row['cycle_number']} ${(row['timestamp'] ?? '').toString().split('T').last}',
                  style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _outcomes(AppState state) {
    return Panel(
      title: 'Outcomes',
      trailing: Text(
        'win rate ${(state.winRate * 100).toStringAsFixed(0)}% '
        '(${state.outcomeCount})',
        style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
      ),
      child: Column(
        children: state.outcomes.take(6).map((outcome) {
          final color = outcome.outcome > 0
              ? AppTheme.buy
              : outcome.outcome < 0
                  ? AppTheme.sell
                  : AppTheme.hold;
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: Row(
              children: [
                SizedBox(
                  width: 52,
                  child: Text(outcome.label,
                      style: TextStyle(
                          color: color,
                          fontSize: 11.5,
                          fontWeight: FontWeight.w700)),
                ),
                Expanded(
                  child: Text(
                    '${outcome.signal} · ${outcome.pnlBps.toStringAsFixed(1)} bps',
                    style: const TextStyle(fontSize: 11.5),
                  ),
                ),
                Text('cycle ${outcome.cycleNumber}',
                    style:
                        const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }
}
