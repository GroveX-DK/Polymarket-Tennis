# Can you beat Polymarket on ATP tennis? — Backtest report

**Date:** 2026-07-02
**Question:** Build a model to predict ATP matches, compare against historical Polymarket
prices, and test whether any strategy earns positive PnL.

**Answer: mostly no — with one modest exception.** Polymarket's ATP match-winner markets
are, near match start, as sharp as the multi-bookmaker consensus and essentially perfectly
calibrated; no model-vs-market strategy survives bias controls. The exception is a
**favorite–longshot bias in early prices**: buying strong favorites (80–97¢) ~24h before
match start, confirmed by the Elo model (p ≥ 0.70), earned **+5.3% ROI (+7.8 units over
147 bets, 95% CI [+0.3%, +9.7%], hit rate 93.2%)** at 1¢ slippage, positive in both
half-years of the sample. See "The one strategy that survived" below — including why its
statistical significance should be discounted somewhat.

---

## Data

| Source | Contents |
|---|---|
| tennis-data.co.uk (`data_raw/td_*.parquet`, `2026.xlsx`) | 70,812 ATP matches 2000–2026 with results and bookmaker odds (Pinnacle, Bet365, market average) |
| Polymarket Gamma API | 17,452 closed tennis events; 16,229 match-winner (moneyline) markets |
| Polymarket CLOB API | Price history for every matched market; snapshots taken 24h / 6h / 1h / 0h before scheduled start |

**Matched sample:** 2,311 Polymarket ATP tour-level match markets (May 2025 – Jun 2026)
joined to tennis-data results by player surname + date; 2,221 clean after dropping
retirements and resolution mismatches (99.4% resolution agreement pre-filter).

## Model

Surface-blended Elo (60% overall / 40% surface, K = 250/(n+5)^0.4), trained
chronologically from 2000, with a calibration scale (s = 0.85) fitted on 2005–2020.
Holdout validation 2021–2024 (n = 9,149):

| Predictor | log-loss | accuracy |
|---|---|---|
| Calibrated Elo | 0.6264 | 63.8% |
| Pinnacle closing (vig-free) | 0.5918 | 67.3% |

Calibration curve is clean (predicted ≈ actual in every decile) — the model is honest,
just less informed than the market, as expected for a rating-only model.

## Who is sharper on the matched Polymarket sample?

Log-loss on identical matches (lower = sharper):

| Predictor | 24h out | 1h out | at start |
|---|---|---|---|
| Polymarket price | 0.5935 | 0.5810 | **0.5781** |
| Bookmaker average (closing)* | 0.5852 | 0.5800 | 0.5791 |
| Calibrated Elo | 0.6172 | 0.6104 | 0.6098 |

*Bookmaker "closing" odds benefit from look-ahead at earlier horizons — see caveats.

By match start **Polymarket is sharper than the bookmaker consensus**, and a logistic
regression gives the PM price a weight of ~1.0 with Elo adding nothing out-of-sample
(2026 test log-loss got *worse* when adding Elo: 0.5909 vs 0.5904).

## Strategy backtests (flat $1 per bet, 1¢ slippage, 95% bootstrap CI)

| Strategy | n | ROI | 95% CI | Verdict |
|---|---|---|---|---|
| Elo edge > 3% @1h | 1,473 | −5.7% | [−13.1, +1.7] | loses |
| Elo edge > 8% @1h | 749 | −6.5% | [−17.6, +5.2] | loses |
| Books vs PM, edge > 3% @1h (capped) | 42 | −3.0% | [−64.9, +89.6] | nothing |
| Books vs PM, edge > 3% @24h (capped)* | 173 | +8.6% | [−8.3, +25.7] | not significant, not implementable* |
| Buy PM favorites @24h | 827 | −0.9% | [−5.1, +3.4] | flat |
| Buy PM longshots @24h | 827 | −7.1% | [−18.4, +4.3] | loses (mild longshot bias) |
| Momentum (follow 24h→1h move) | 394 | −5.9% | [−14.2, +2.6] | loses |
| Contrarian (fade 24h→1h move) | 394 | +3.1% | [−11.5, +18.1] | not significant |

No confidence interval clears zero.

## The look-ahead trap (important negative result)

A naive backtest of "bet with the bookmaker consensus when Polymarket disagrees" showed
**+55–69% ROI** at large divergence thresholds. This is an artifact. Spot checks of the
largest "edges" proved tennis-data's 2025/26 odds are contaminated with post-result
information on some rows:

- **Zverev vs Auger-Aliassime, ATP Finals 2025-11-14** — tennis-data: FAA priced 1.19
  (84%). Reality: Zverev was the −173 favorite (~63%), exactly matching Polymarket's
  stable 0.62 on $1.45M volume. FAA won; the recorded odds contain the result.
- **Bublik vs Tsitsipas, Madrid 2026-04-25** — tennis-data: Tsitsipas 1.14 (88%).
  Reality: Bublik was the ~56% pre-match favorite, matching Polymarket. Tsitsipas won.

After capping divergences at 0.15, the phantom profit collapses to the insignificant
+8.6% above — and even that compares *closing* odds to PM prices 24h earlier, which is
not a tradeable signal (you'd need tomorrow's closing line today).

## The one strategy that survived: Elo-confirmed strong favorites, bought early

Early Polymarket prices (24h+ before start) show the classic favorite–longshot bias:
in the 24h calibration table, favorites priced 0.80–0.90 won 89.4% of the time and those
above 0.90 won 93.9% — both above their price. Longshots were correspondingly overpriced
(buying them lost −7.1%). The bias vanishes by 1h before start as the market matures.

**Spec** (`scripts/final_strategy.py`): 24h before scheduled start, buy the favorite if
its last-trade price is in [0.80, 0.97], market volume ≥ $1k, last trade ≤ 6h old, and
the calibrated Elo model also gives that side ≥ 0.70. Flat $1 per bet, hold to resolution.

| Fill assumption | n | hit | ROI | 95% CI | total PnL |
|---|---|---|---|---|---|
| Maker (0¢) | 147 | 93.2% | +6.5% | [+1.6, +10.9] | +9.6 units |
| Taker (1¢ slip) | 147 | 93.2% | **+5.3%** | [+0.3, +9.7] | **+7.8 units** |
| Taker (2¢ slip) | 147 | 93.2% | +4.1% | [−0.6, +8.5] | +6.1 units |

Stability: +3.2% ROI in 2025H2, +6.4% in 2026H1; every Elo-filter setting in the sweep
(none → 0.80) was positive (+3.7% to +6.0%); losses (10 of 147) are scattered, plausible
upsets — ~18 would be expected if prices were fair.

**Discount factors, stated plainly:** the Elo cutoff (0.70) was chosen from a small sweep,
this was one of several strategy families examined (multiple-testing risk), the CI only
just clears zero at taker fills, and last-trade prices 24h out may understate what a real
fill would cost even in $1k+ markets. Treat +5% as an optimistic point estimate of a real
but thin edge that exists only in early, less liquid trading — roughly $50 per week of
tour play at $1k total stakes/week. Equity curve: `final_strategy_equity.png`.

## Honest interpretation

1. **Polymarket ATP moneylines are efficient near match start.** Prices 1h out are as
   sharp as the multi-book consensus and perfectly calibrated; sharper than books by start.
2. **A rating-only model can't beat them head-on.** Calibrated Elo bleeds roughly the
   spread (−5 to −7% ROI) at every threshold.
3. **Early prices are the soft spot.** 24h+ out, log-loss is 0.594 vs 0.585 for the
   eventual consensus, longshots are overbought, and strong favorites are underpriced —
   the source of the one surviving strategy above. Blend/convergence strategies that
   looked even better turned out to be artifacts of stale last-trade prints (placeholder
   0.50 prices on not-yet-traded markets); always check both time halves.
4. **Beware tennis-data 2025/26 odds** for any market-comparison research — some rows
   are recorded in-play or post-result (verified against contemporaneous news).

## Micro-stats + context model (added 2026-07-07)

Follow-up question: how much of the Elo-vs-market gap can serve/return
micro-stats and context features close? (`scripts/microstats_model.py`, data via
`scripts/fetch_tml.py` — Sackmann's repo is offline; TML-Database mirrors the
same per-match stats schema through the current season.)

Features (all computed strictly pre-match, EWMA halflife 25 matches, overall +
per-surface): serve/return points won %, ace/DF rates, hold %, BP save %;
context: best-of-5, round, indoor, home country, retirement/walkover risk;
base: overall + surface Elo, rank, rank points, age, height. Model:
HistGradientBoosting trained on 2000–2020, both orientations (symmetric).

| Predictor (holdout 2021–2024, n = 9,005, identical rows) | log-loss | accuracy |
|---|---|---|
| Calibrated Elo (report model) | 0.6232 | 63.6% |
| GBM, Elo features only (ablation) | 0.6253 | 63.7% |
| **GBM, micro-stats + context** | **0.6130** | **64.9%** |

The improvement (+0.0102 log-loss, bootstrap 95% CI [+0.0071, +0.0131]) is real
but closes only ~a third of the gap to the market. Top non-Elo features: **age
difference** (aging curves), rank points, best-of-5, return points won, home
advantage.

On the 2025 Polymarket matched sample (n = 793 with prices at start) the story
doesn't change: PM 0.5631, GBM 0.5938, Elo 0.5905 — the market stays ~0.03
ahead, and a chronological two-fold logistic blend gives the GBM **zero-to-negative
weight on top of the PM price** (same conclusion as for Elo: no added
information). One suggestive positive: as the confirmation filter for the 24h
favorites strategy on this subsample, GBM ≥ 0.70 beat Elo ≥ 0.70 (+9.9% vs
+7.4% ROI on 41 bets) — but that's a one-bet difference, far from significant.

Practical note: the live bot keeps the Elo filter — TML stats data lags weeks
behind (its 2026 file ends in January), so a GBM filter can't be fed reliably
in production, and it adds sklearn/numpy to the Pi's footprint for an
unproven filter gain.

## Live dry run (added 2026-07-07)

The surviving strategy is deployed as a paper-trading bot in `pi_bot/` —
pure Python stdlib (no pandas/numpy), ~100 MB peak memory, built to run 24/7
on a Raspberry Pi with 1 GB RAM. It polls the Gamma API every 30 minutes,
enters qualifying favorites 18–30h before start (ATP tour-level only, via a
tournament-calendar gate), settles from on-chain resolutions into
`pi_bot/data/paper_trades.sqlite`, and refreshes its Elo ratings weekly from
tennis-data results. See `pi_bot/README.md` for systemd/cron setup.

## Reproduce

```
python scripts/harvest_events.py   # Gamma API -> events_tennis.jsonl (~15 min)
python scripts/build_elo.py        # Elo + calibration -> elo_predictions.parquet
python scripts/match_markets.py    # join PM markets to results -> matched.parquet
python scripts/fetch_prices.py     # CLOB price snapshots (~5 min)
python scripts/backtest.py         # naive backtest (shows the phantom edge)
python scripts/backtest2.py        # bias-controlled backtest (the honest one)
python scripts/blend_test.py       # does anything add info beyond the PM price?
python scripts/backtest3.py        # blend strategies (exposed as stale-print artifact)
python scripts/favorites.py        # favorite-longshot bias buckets
python scripts/final_strategy.py   # the surviving strategy + equity curve
python scripts/make_charts.py      # report_charts.png
```
