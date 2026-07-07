# pi_bot — 24/7 Polymarket ATP dry-run paper trader

Paper-trades the one strategy that survived the 2025–2026 backtest (see
`../REPORT.md`): **buy Polymarket favorites priced 0.80–0.97 about 24h before
match start, when the calibrated Elo model also gives that side ≥ 0.70**, on
ATP tour-level singles only, volume ≥ $1k, tight book.
Backtest: +5.3% ROI, 147 bets, 95% CI [+0.3%, +9.7%] at 1¢ taker slippage.

Built for a Raspberry Pi with 1 GB RAM: **pure Python stdlib** (no pandas,
numpy, or any pip install), ~100 MB peak / far less at idle, one HTTPS poll
of the Gamma API every 30 minutes.

## Files

| file | role |
|---|---|
| `bot.py` | the 24/7 loop: scan → paper-enter → resolve → log PnL |
| `tennis_elo.py` | pure-python Elo engine + player-name matching (mirrors `scripts/build_elo.py`) |
| `update_ratings.py` | weekly: download tennis-data results, refresh `data/matches.csv.gz` + tour calendar |
| `export_history.py` | one-time, on the PC (needs pandas): seed `data/matches.csv.gz` + `data/tournaments.json` |
| `data/paper_trades.sqlite` | the paper ledger (created automatically) |

Only *results* are taken from tennis-data — its 2025/26 odds columns are
known-contaminated (REPORT.md) and never read.

## Run

```bash
python3 bot.py            # loop forever (30-min cycles)
python3 bot.py --once     # single cycle (testing / cron-driven use)
python3 bot.py report     # ledger summary: hit rate, ROI, last 15 trades
```

A paper entry is logged as `PAPER BUY <player> @ <ask> ...` and settled
automatically a few hours after the match from the market's on-chain
resolution (`won` / `lost` / `void` for 50-50 refunds).

## Raspberry Pi deployment

Copy the `pi_bot/` folder (with `data/matches.csv.gz` + `data/tournaments.json`
already seeded) to the Pi, then:

```ini
# /etc/systemd/system/tennisbot.service
[Unit]
Description=Polymarket ATP dry-run bot
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/home/pi/pi_bot
ExecStart=/usr/bin/python3 /home/pi/pi_bot/bot.py
Restart=always
RestartSec=60
MemoryMax=300M

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now tennisbot
journalctl -u tennisbot -f          # watch it live
```

Weekly ratings refresh (tennis-data updates roughly weekly; the bot reloads
the file automatically when it changes):

```
# crontab -e
17 6 * * 1  cd /home/pi/pi_bot && /usr/bin/python3 update_ratings.py >> data/update.log 2>&1
```

## Strategy parameters (top of `bot.py`)

| param | value | why |
|---|---|---|
| `PRICE_LO..PRICE_HI` | 0.80–0.97 | validated favorite bucket |
| `ELO_MIN` | 0.70 | model confirmation (backtest sweep optimum) |
| `HRS_MIN..HRS_MAX` | 18–30h | entry window around the validated 24h snapshot |
| `VOL_MIN` | $1,000 | liquidity floor from the backtest |
| `MAX_SPREAD` | 0.05 | live proxy for the backtest's "fresh trade ≤ 6h" filter |
| `STAKE` | $1 flat | same as backtest units |

Expect roughly **2–3 qualifying bets per week** (147 bets over 13 backtest
months), clustering around big tournaments. Judge the dry run on ≥ 100
settled bets — about a season of tour play; the edge
is thin (+5% point estimate) and REPORT.md lists the reasons to discount it
(mild cutoff tuning, multiple testing, stale 24h prints).
