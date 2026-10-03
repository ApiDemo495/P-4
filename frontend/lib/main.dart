import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'screens/dashboard_screen.dart';
import 'state/app_state.dart';
import 'theme.dart';

/// DROSOPHILA TRADER v2.0 — Flutter client.
///
/// Run against a local engine:
///
///     flutter run -d chrome --dart-define=API_BASE=http://localhost:8000
///
/// The client is a pure view of the backend: it consumes `/ws/signals` plus the
/// REST mirrors and never computes a signal of its own.
void main() {
  runApp(const DrosophilaApp());
}

class DrosophilaApp extends StatelessWidget {
  const DrosophilaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AppState()..start(),
      child: MaterialApp(
        title: 'Drosophila Trader v2.0',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.dark,
        home: const DashboardScreen(),
      ),
    );
  }
}
