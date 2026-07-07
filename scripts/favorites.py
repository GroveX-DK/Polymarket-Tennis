"""Favorite-longshot bias on Polymarket ATP: buy favorites by price bucket.
Filters: exclude placeholder prices (0.48-0.52 at 24h with big later jump is
handled by bucket >= 0.70 anyway), require volume, fresh snapshot trades.
Stability: both time halves, multiple horizons, slippage 0 (maker) and 1c (taker).
"""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(3)

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[(df["td_comment"] == "Completed") & (df["winner_final_price"] > 0.5)]
df = df.dropna(subset=["p0_24h"]).copy()
df["y0"] = df["winner_is_outcome0"].astype(float)
half = df["game_start_ts"].quantile(0.5)

def summ(pnl):
    idxs = rng.integers(0, len(pnl), (4000, len(pnl)))
    rois = pnl.to_numpy()[idxs].mean(1)
    return pnl.mean(), *np.percentile(rois, [2.5, 97.5])

def fav_bets(t, snap, lo, hi, slip, min_vol):
    tt = t.dropna(subset=[snap])
    tt = tt[tt["volume"].fillna(0) >= min_vol]
    # snapshot freshness: last trade within 6h of cutoff
    tcol = snap + "_t"
    lag = {"p0_24h": 24 * 3600, "p0_6h": 6 * 3600, "p0_1h": 3600, "p0_0h": 0}[snap]
    age = (tt["game_start_ts"].astype("int64") / 1e9 - lag - tt[tcol]) / 3600
    tt = tt[age <= 6]
    rows = []
    for r in tt.itertuples():
        q0 = getattr(r, snap)
        for (q, y) in [(q0, r.y0), (1 - q0, 1 - r.y0)]:
            if lo <= q <= hi:
                qe = q + slip
                rows.append(((y - qe) / qe, y, qe))
                break
    return pd.DataFrame(rows, columns=["pnl", "won", "q"])

print(f"{'snap':7} {'bucket':>11} {'slip':>5} {'minvol':>6} | {'n':>4} {'hit':>6} {'ROI%':>7} {'95% CI':>16}")
for snap in ["p0_24h", "p0_6h", "p0_1h"]:
    for lo, hi in [(0.70, 0.97), (0.75, 0.97), (0.80, 0.97), (0.85, 0.97)]:
        for slip in [0.0, 0.01]:
            b = fav_bets(df, snap, lo, hi, slip, 1000)
            if len(b) < 30:
                continue
            roi, l, h = summ(b["pnl"])
            print(f"{snap:7} [{lo:.2f},{hi:.2f}] {slip:5.2f} {1000:6d} | {len(b):4d} {b['won'].mean():6.3f} {roi*100:7.1f} [{l*100:6.1f},{h*100:6.1f}]")

print("\n=== stability: halves (slip=0.01, minvol=1000) ===")
for snap in ["p0_24h", "p0_6h", "p0_1h"]:
    for lo, hi in [(0.75, 0.97), (0.80, 0.97)]:
        for label, t in [("early", df[df["game_start_ts"] <= half]), ("late", df[df["game_start_ts"] > half])]:
            b = fav_bets(t, snap, lo, hi, 0.01, 1000)
            if len(b) < 20:
                continue
            roi, l, h = summ(b["pnl"])
            print(f"{snap:7} [{lo:.2f},{hi:.2f}] {label:5} | n={len(b):4d} hit={b['won'].mean():.3f} ROI={roi*100:6.1f}% [{l*100:6.1f},{h*100:6.1f}]")
