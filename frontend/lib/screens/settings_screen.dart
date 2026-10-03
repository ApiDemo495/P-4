import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';

/// Settings: key testing, local model, brain control, news.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.state});

  final AppState state;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  /// Three boxes per provider: slot 1 is the primary (always preferred),
  /// slots 2-3 are temporary stand-ins the engine fails over to.
  static const _providers = ['gemini', 'github', 'cryptopanic', 'newsapi'];
  final Map<String, List<TextEditingController>> _keys = {
    for (final p in _providers)
      p: List.generate(3, (_) => TextEditingController()),
  };
  bool _persist = false;
  final Map<String, String> _status = {};

  @override
  void dispose() {
    for (final controllers in _keys.values) {
      for (final c in controllers) {
        c.dispose();
      }
    }
    super.dispose();
  }

  Future<void> _test(String name, int slot) async {
    setState(() => _status[name] = 'testing slot $slot…');
    final result = await widget.state.api
        .testAgent(name, _keys[name]![slot - 1].text.trim());
    final valid = result?['valid'] == true;
    setState(() => _status[name] = valid
        ? '✅ slot $slot: ${result?['detail'] ?? 'valid'}'
        : '❌ slot $slot: ${result?['error'] ?? 'not reachable'}');
  }

  Future<void> _save() async {
    for (final provider in _providers) {
      final controllers = _keys[provider]!;
      for (var slot = 1; slot <= controllers.length; slot++) {
        final value = controllers[slot - 1].text.trim();
        if (value.isEmpty) continue;
        await widget.state.api
            .saveKey(provider, value, persist: _persist, slot: slot);
      }
    }
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Keys saved')),
      );
    }
    await widget.state.refreshAgents();
  }

  @override
  Widget build(BuildContext context) {
    final state = widget.state;
    final brain = state.brainStatus;
    final local = state.agentStatus['local'];

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        Panel(
          title: 'API keys',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _keyField('gemini',
                  'Gemini API keys — primary + 2 stand-ins (all optional)', 'AIza…'),
              _keyField('github', 'GitHub PATs (optional)', 'github_pat_…'),
              _keyField('cryptopanic', 'CryptoPanic keys (optional)', 'token'),
              _keyField('newsapi', 'NewsAPI keys (optional)', 'key'),
              const Text(
                'Slot 1 is always preferred. When it is rejected, rate limited or '
                'errors the engine switches to slot 2, then 3, and returns to slot 1 '
                'as soon as its cooldown ends.',
                style: TextStyle(color: AppTheme.textMuted, fontSize: 11),
              ),
              Row(
                children: [
                  Checkbox(
                    value: _persist,
                    onChanged: (v) => setState(() => _persist = v ?? false),
                  ),
                  const Expanded(
                    child: Text(
                      'persist to .env (git-ignored) — otherwise keys live in '
                      'memory for this session only',
                      style: TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
                    ),
                  ),
                ],
              ),
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton(onPressed: _save, child: const Text('Save')),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Panel(
          title: 'Local AI model',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                local == null
                    ? 'No model loaded'
                    : '${local['status']} · ${local['model'] ?? 'no model'} · '
                        '${local['parameter_count'] ?? ''}',
                style: const TextStyle(fontSize: 12.5),
              ),
              const SizedBox(height: 6),
              const Text(
                'Upload a .gguf or .onnx file from the dashboard (Settings → '
                'Upload). Files are validated by magic bytes and size before '
                'loading, then test-inferred, then monitored every 60 s.',
                style: TextStyle(color: AppTheme.textMuted, fontSize: 11.5),
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  OutlinedButton(
                    onPressed: state.loadStubModel,
                    child: const Text('Load dev stub'),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton(
                    onPressed: () async {
                      await state.api.localModelUnload();
                      await state.refreshAgents();
                    },
                    child: const Text('Unload'),
                  ),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Panel(
          title: 'Brain connection',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${brain['status'] ?? '—'} · ${brain['message'] ?? ''}',
                style: const TextStyle(fontSize: 12.5),
              ),
              StatRow(
                label: 'matrix',
                value: brain['matrix']?['shape']?.toString() ?? '—',
              ),
              StatRow(
                label: 'checksum',
                value: brain['matrix']?['checksum']?.toString() ?? '—',
              ),
              StatRow(label: 'gain', value: brain['gain']?.toString() ?? '—'),
              StatRow(
                label: 'neuPrint',
                value: brain['health']?['neuprint_live'] == true
                    ? 'reachable'
                    : 'unreachable (fallback matrix)',
              ),
              const SizedBox(height: 10),
              OutlinedButton(
                onPressed: state.reconnectBrain,
                child: const Text('Force reconnect'),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Panel(
          title: 'News',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              StatRow(label: 'coverage', value: state.news?.coverage ?? '—'),
              StatRow(
                label: 'poll age',
                value: state.news?.ageSeconds == null
                    ? '—'
                    : '${state.news!.ageSeconds!.toStringAsFixed(0)}s',
              ),
              const SizedBox(height: 10),
              OutlinedButton(
                onPressed: state.pollNews,
                child: const Text('Poll now'),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _keyField(String name, String label, String hint) {
    final controllers = _keys[name]!;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          for (var slot = 1; slot <= controllers.length; slot++)
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                children: [
                  SizedBox(
                    width: 64,
                    child: Text(
                      slot == 1 ? 'PRIMARY' : 'BACKUP ${slot - 1}',
                      style: const TextStyle(
                          color: AppTheme.textMuted, fontSize: 10),
                    ),
                  ),
                  Expanded(
                    child: TextField(
                      controller: controllers[slot - 1],
                      obscureText: true,
                      decoration: InputDecoration(
                        hintText: slot == 1 ? hint : 'optional',
                        isDense: true,
                        filled: true,
                        fillColor: AppTheme.surfaceAlt,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: const BorderSide(color: AppTheme.border),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton(
                    onPressed: () => _test(name, slot),
                    child: const Text('Test'),
                  ),
                ],
              ),
            ),
          if (_status[name] != null) ...[
            const SizedBox(height: 4),
            Text(_status[name]!,
                style: const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
          ],
        ],
      ),
    );
  }
}
