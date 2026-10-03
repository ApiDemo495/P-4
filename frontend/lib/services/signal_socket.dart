import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

/// A single WebSocket event from `/ws/signals`.
class SignalEvent {
  const SignalEvent(this.type, this.data);

  final String type;
  final Map<String, dynamic> data;
}

/// Resilient client for the one real-time channel of the engine.
///
/// Reconnects with exponential backoff (1 s, 2 s, 4 s … capped at 15 s) and
/// replays the server's `HELLO` snapshot on every reconnect, so the UI never
/// has to reconstruct state by itself.
class SignalSocket {
  SignalSocket(this.baseUrl);

  final String baseUrl;
  final _controller = StreamController<SignalEvent>.broadcast();

  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _subscription;
  Timer? _retryTimer;
  Timer? _silenceTimer;
  DateTime _lastFrame = DateTime.now();
  int _attempt = 0;
  bool _closed = false;

  /// The engine streams the crowd twice a second, so a socket that stays open
  /// without a frame for this long is a half-open proxy connection (the
  /// Codespaces forwarder keeps our side alive after the backend stalled).
  /// Round X: close it, which reconnects and replays `HELLO`.
  static const Duration silenceLimit = Duration(seconds: 15);

  Stream<SignalEvent> get events => _controller.stream;

  bool get connected => _channel != null;

  Uri get _uri {
    final uri = Uri.parse(baseUrl);
    final scheme = uri.scheme == 'https' ? 'wss' : 'ws';
    return uri.replace(scheme: scheme, path: '/ws/signals', query: '');
  }

  void connect() {
    if (_closed) return;
    _teardown();
    try {
      final channel = WebSocketChannel.connect(_uri);
      _channel = channel;
      _subscription = channel.stream.listen(
        _onData,
        onDone: _scheduleReconnect,
        onError: (_) => _scheduleReconnect(),
        cancelOnError: true,
      );
      _controller.add(const SignalEvent('SOCKET_OPEN', {}));
      _attempt = 0;
      _lastFrame = DateTime.now();
      _silenceTimer?.cancel();
      _silenceTimer = Timer.periodic(const Duration(seconds: 5), (_) {
        if (_channel == null) return;
        if (DateTime.now().difference(_lastFrame) > silenceLimit) {
          _controller.add(const SignalEvent('SOCKET_SILENT', {}));
          _scheduleReconnect();
        }
      });
    } catch (_) {
      _scheduleReconnect();
    }
  }

  void _onData(dynamic raw) {
    _lastFrame = DateTime.now();
    try {
      final decoded = jsonDecode(raw.toString());
      if (decoded is! Map) return;
      final type = (decoded['type'] ?? '').toString();
      final data = decoded['data'];
      _controller.add(
        SignalEvent(type, data is Map ? Map<String, dynamic>.from(data) : {}),
      );
    } catch (_) {
      // a malformed frame must never kill the stream
    }
  }

  void _scheduleReconnect() {
    if (_closed) return;
    _teardown();
    _controller.add(const SignalEvent('SOCKET_CLOSED', {}));
    _attempt = (_attempt + 1).clamp(1, 4);
    final delay = Duration(seconds: (1 << (_attempt - 1)).clamp(1, 15));
    _retryTimer?.cancel();
    _retryTimer = Timer(delay, connect);
  }

  void _teardown() {
    _silenceTimer?.cancel();
    _silenceTimer = null;
    _subscription?.cancel();
    _subscription = null;
    _channel?.sink.close();
    _channel = null;
  }

  /// Ask the engine to queue an asset switch (applies at the next cycle).
  void switchAsset(String asset) {
    _channel?.sink.add(jsonEncode({'action': 'switch_asset', 'asset': asset}));
  }

  void ping() => _channel?.sink.add(jsonEncode({'action': 'ping'}));

  void dispose() {
    _closed = true;
    _retryTimer?.cancel();
    _teardown();
    _controller.close();
  }
}
