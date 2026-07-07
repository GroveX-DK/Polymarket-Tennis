# Polymarket Tennis

Can a model beat Polymarket's ATP match-winner markets? Mostly no — with one
modest exception that survived all bias controls, now running as a 24/7 paper
trader.

- **[REPORT.md](REPORT.md)** — the full backtest: 2,221 Polymarket ATP markets
  (May 2025 – Jun 2026) vs a calibrated surface-blended Elo model and bookmaker
  odds. Headline: markets are efficient near match start, but early prices
  (~24h out) underprice strong favorites. Buying 0.80–0.97¢ favorites with Elo
  confirmation ≥ 0.70 earned **+5.3% ROI over 147 bets** (95% CI [+0.3%, +9.7%]
  at 1¢ slippage).
- **[pi_bot/](pi_bot/)** — the dry-run bot: pure Python stdlib, ~100 MB peak
  memory, built for a Raspberry Pi with 1 GB RAM. Polls the Gamma API every
  30 min, paper-trades the strategy, settles from on-chain resolutions, and
  refreshes its Elo ratings weekly from tennis-data.co.uk results.
- **[scripts/](scripts/)** — the research pipeline (harvest → Elo → match →
  price snapshots → backtests). Reproduction steps at the bottom of REPORT.md.

Data note: `data_raw/polymarket/events_tennis.jsonl` (109 MB) is not committed;
regenerate it with `python scripts/harvest_events.py`. Beware tennis-data's
2025/26 *odds* columns — some rows are contaminated with in-play/post-result
information (verified in REPORT.md); results are fine.
