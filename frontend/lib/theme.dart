import 'dart:ui' show FontFeature;

import 'package:flutter/material.dart';

/// Dashboard palette - identical to `backend/web/styles.css`, so the two
/// presentation layers look like the same product.
class AppTheme {
  static const Color background = Color(0xFF070B12);
  static const Color surface = Color(0xFF0F1826);
  static const Color surfaceAlt = Color(0xFF141F31);
  static const Color border = Color(0xFF1E2C42);
  static const Color textPrimary = Color(0xFFE8EEF7);
  static const Color textMuted = Color(0xFF8A9BB4);
  static const Color buy = Color(0xFF22C55E);
  static const Color sell = Color(0xFFEF4444);
  static const Color hold = Color(0xFF94A3B8);
  static const Color accent = Color(0xFF38BDF8);
  static const Color warning = Color(0xFFFBBF24);
  static const Color emergency = Color(0xFFF97316);

  static ThemeData get dark {
    final base = ThemeData.dark(useMaterial3: true);
    return base.copyWith(
      scaffoldBackgroundColor: background,
      colorScheme: base.colorScheme.copyWith(
        primary: accent,
        secondary: buy,
        surface: surface,
        error: sell,
      ),
      cardColor: surface,
      dividerColor: border,
      textTheme: base.textTheme.apply(
        bodyColor: textPrimary,
        displayColor: textPrimary,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: surface,
        elevation: 0,
        centerTitle: false,
      ),
    );
  }

  static Color forSignal(String signal) {
    switch (signal) {
      case 'BUY':
        return buy;
      case 'SELL':
        return sell;
      default:
        return hold;
    }
  }

  static Color forValue(double value) {
    if (value > 0.15) return buy;
    if (value < -0.15) return sell;
    return textMuted;
  }
}

/// A uniformly styled dark panel.
class Panel extends StatelessWidget {
  const Panel({
    super.key,
    required this.child,
    this.title,
    this.trailing,
    this.padding = const EdgeInsets.all(16),
  });

  final Widget child;
  final String? title;
  final Widget? trailing;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border),
      ),
      padding: padding,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (title != null) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  title!.toUpperCase(),
                  style: const TextStyle(
                    fontSize: 11,
                    letterSpacing: 1.4,
                    color: AppTheme.textMuted,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (trailing != null) trailing!,
              ],
            ),
            const SizedBox(height: 12),
          ],
          child,
        ],
      ),
    );
  }
}

/// Signed horizontal bar used for formula values and hedge metrics.
class SignedBar extends StatelessWidget {
  const SignedBar({
    super.key,
    required this.value,
    this.min = -1,
    this.max = 1,
    this.height = 8,
    this.color,
  });

  final double value;
  final double min;
  final double max;
  final double height;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final span = (max - min).abs();
    final clipped = value.clamp(min, max);
    final zero = (-min) / span;
    final fraction = (clipped - min) / span;
    final left = fraction < zero ? fraction : zero;
    final width = (fraction - zero).abs();
    return LayoutBuilder(
      builder: (context, constraints) {
        final total = constraints.maxWidth;
        return SizedBox(
          height: height,
          child: Stack(
            children: [
              Container(
                decoration: BoxDecoration(
                  color: AppTheme.background,
                  borderRadius: BorderRadius.circular(height / 2),
                ),
              ),
              Positioned(
                left: total * zero - 0.5,
                top: 0,
                bottom: 0,
                child: Container(width: 1, color: AppTheme.border),
              ),
              Positioned(
                left: total * left,
                width: (total * width).clamp(0.0, total),
                top: 0,
                bottom: 0,
                child: Container(
                  decoration: BoxDecoration(
                    color: color ?? AppTheme.forValue(value),
                    borderRadius: BorderRadius.circular(height / 2),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

/// Compact "label: value" row.
class StatRow extends StatelessWidget {
  const StatRow({
    super.key,
    required this.label,
    required this.value,
    this.color = AppTheme.textPrimary,
    this.mono = true,
  });

  final String label;
  final String value;
  final Color color;
  final bool mono;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: AppTheme.textMuted, fontSize: 12)),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 12.5,
              fontFeatures: mono ? const [FontFeature.tabularFigures()] : null,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
