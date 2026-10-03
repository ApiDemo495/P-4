import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';

/// News bar: latest headline, source tier, poll age, NIV/SMD and coverage.
class NewsCard extends StatelessWidget {
  const NewsCard({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final news = state.news;
    final signal = state.signal;
    final headline = news?.headline ??
        (signal?.news['latest_headline']?.toString() ?? 'No headlines available');
    final source = news?.source ?? signal?.news['source']?.toString() ?? '';
    final tier = news?.tier ?? (signal?.news['tier'] as num?)?.toInt() ?? 0;
    final niv = signal?.formulas['NIV'] ?? news?.niv ?? 0;
    final smd = signal?.formulas['SMD'] ?? news?.smd ?? 0;
    final age = news?.ageSeconds ?? (signal?.news['last_poll_seconds_ago'] as num?);

    return Panel(
      title: 'News',
      trailing: Text(
        'coverage ${news?.coverage ?? '—'}',
        style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(headline,
              style: const TextStyle(fontSize: 13.5, height: 1.35),
              maxLines: 3,
              overflow: TextOverflow.ellipsis),
          const SizedBox(height: 8),
          Text(
            [
              if (source.isNotEmpty) '$source · Tier $tier',
              if (age != null) 'polled ${age.toStringAsFixed(0)}s ago',
            ].join(' · '),
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 11),
          ),
          const SizedBox(height: 10),
          StatRow(
            label: 'NIV (news impact velocity)',
            value: niv.toStringAsFixed(3),
            color: AppTheme.forValue(niv),
          ),
          StatRow(
            label: 'SMD (sentiment vs tape)',
            value: smd.toStringAsFixed(3),
            color: AppTheme.forValue(smd),
          ),
          const SizedBox(height: 10),
          ...?state.news?.items.take(3).map(
                (item) => Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        margin: const EdgeInsets.only(top: 2, right: 6),
                        padding:
                            const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                        decoration: BoxDecoration(
                          color: AppTheme.surfaceAlt,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text('T${item.tier}',
                            style: const TextStyle(
                                fontSize: 9.5, color: AppTheme.textMuted)),
                      ),
                      Expanded(
                        child: Text(
                          item.headline,
                          style: const TextStyle(
                              fontSize: 11.5, color: AppTheme.textMuted),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      Text(
                        item.sentiment.toStringAsFixed(2),
                        style: TextStyle(
                          fontSize: 11,
                          color: AppTheme.forValue(item.sentiment),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
        ],
      ),
    );
  }
}
