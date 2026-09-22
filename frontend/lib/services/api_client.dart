import 'dart:convert';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;

import '../models/signal.dart';

/// REST mirror of the dashboard API.
///
/// The WebSocket is the primary channel; this client exists for the initial
/// snapshot, the Formula Explorer metadata and the Settings screen.
class ApiClient {
  ApiClient({String? baseUrl})
      : baseUrl = (baseUrl ?? resolvedBase).replaceAll(RegExp(r'/+$'), '');

  /// Optional compile-time override:
  ///
  ///     flutter run --dart-define=API_BASE=http://192.168.1.20:8000
  ///
  /// When it is not supplied the client talks to **the origin it was served
  /// from**, which is what makes `frontend/build/web` served by the backend at
  /// `/flutter` work with no configuration at all (and no CORS).
  static const String configuredBase =
      String.fromEnvironment('API_BASE', defaultValue: '');

  static String get resolvedBase {
    if (configuredBase.isNotEmpty) return configuredBase;
    if (kIsWeb) return Uri.base.origin; // e.g. https://<name>-8000.app.github.dev
    return 'http://localhost:8000';
  }

  /// Kept for compatibility with earlier call sites.
  static String get defaultBaseUrl => resolvedBase;

  final String baseUrl;

  Uri _uri(String path, [Map<String, String>? query]) =>
      Uri.parse('$baseUrl$path').replace(queryParameters: query);

  Future<Map<String, dynamic>?> _getJson(String path,
      [Map<String, String>? query]) async {
    try {
      final response =
          await http.get(_uri(path, query)).timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) return null;
      final decoded = jsonDecode(response.body);
      return decoded is Map<String, dynamic> ? decoded : null;
    } catch (_) {
      return null;
    }
  }

  Future<Map<String, dynamic>?> _postJson(String path,
      [Map<String, dynamic>? body]) async {
    try {
      final response = await http
          .post(
            _uri(path),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode(body ?? const {}),
          )
          .timeout(const Duration(seconds: 15));
      if (response.statusCode < 200 || response.statusCode >= 300) return null;
      final decoded = jsonDecode(response.body);
      return decoded is Map<String, dynamic> ? decoded : null;
    } catch (_) {
      return null;
    }
  }

  Future<SystemConfig?> config() async {
    final json = await _getJson('/api/system/config');
    return json == null ? null : SystemConfig.fromJson(json);
  }

  Future<List<FormulaCategory>> formulaCatalog() async {
    final json = await _getJson('/api/formulas');
    if (json == null) return const [];
    return ((json['categories'] as List?) ?? const [])
        .map((c) => FormulaCategory.fromJson(Map<String, dynamic>.from(c as Map)))
        .toList();
  }

  Future<NewsSnapshot?> news() async {
    final json = await _getJson('/api/news', {'limit': '6'});
    return json == null ? null : NewsSnapshot.fromJson(json);
  }

  Future<Map<String, dynamic>?> signalStatus() => _getJson('/api/signal/status');

  Future<Map<String, dynamic>?> health() => _getJson('/api/health');

  Future<Map<String, dynamic>?> agents() => _getJson('/api/agents');

  Future<Map<String, dynamic>?> brainStatus() => _getJson('/api/brain/status');

  Future<Map<String, dynamic>?> formulasLive() => _getJson('/api/formulas/live');

  Future<Map<String, dynamic>?> timings() => _getJson('/api/formulas/timings');

  Future<Map<String, dynamic>?> history({int limit = 20}) =>
      _getJson('/api/signal/history', {'limit': '$limit'});

  Future<Map<String, dynamic>?> outcomes() => _getJson('/api/signal/outcomes');

  Future<Map<String, dynamic>?> switchAsset(String asset) =>
      _postJson('/api/assets/switch', {'asset': asset});

  Future<Map<String, dynamic>?> testAgent(String name, String key) =>
      _postJson('/api/agents/$name/test', {'key': key});

  Future<Map<String, dynamic>?> saveKey(String name, String key,
          {bool persist = false}) =>
      _postJson('/api/settings/keys/$name', {'key': key, 'persist': persist});

  Future<Map<String, dynamic>?> localModelStatus() =>
      _getJson('/api/agents/local/status');

  Future<Map<String, dynamic>?> localModelStub() =>
      _postJson('/api/agents/local/stub?enabled=true');

  Future<Map<String, dynamic>?> localModelUnload() =>
      _postJson('/api/agents/local/unload', {'delete_file': false});

  Future<Map<String, dynamic>?> brainReconnect() =>
      _postJson('/api/brain/reconnect');

  Future<Map<String, dynamic>?> pollNews() => _postJson('/api/news/poll');

  Future<Map<String, dynamic>?> clearEmergency() =>
      _postJson('/api/news/emergency/clear');
}
