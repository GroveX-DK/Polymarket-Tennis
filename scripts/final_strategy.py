"""FINAL STRATEGY: buy strong Polymarket favorites (~24h before start) that the
Elo model confirms. Favorite-longshot bias capture with model confirmation.

Spec (pre-registered before this run):
  entry: price bucket [0.80, 0.97] at 24h snapshot, volume >= 1000,
         snapshot trade within 6h of cutoff
  filter: calibrated Elo prob of same side >= 0.70
  stake: flat $1 cost; slippage 1c taker / 0c maker reported
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
df = df.dropna(subset=["p0_24h", "p_elo_cal"]).copy()
df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])
df = df[df["volume"].fillna(0) >= 1000]
age = (df["game_start_ts"].astype("int64") / 1e9 - 24 * 3600 - df["p0_24h_t"]) / 3600
df = df[age <= 6]
df = df.sort_values("game_start_ts")

def summ(pnl):
    idxs = rng.integers(0, len(pnl), (4000, len(pnl)))
    rois = pnl.to_numpy()[idxs].mean(1)
    return pnl.mean(), *np.percentile(rois, [2.5, 97.5])

def run(t, elo_min, slip, lo=0.80, hi=0.97):
    rows = []
    for r in t.itertuples():
        for (q, y, pe) in [(r.p0_24h, r.y0, r.p_elo0), (1 - r.p0_24h, 1 - r.y0, 1 - r.p_elo0)]:
            if lo <= q <= hi and pe >= elo_min:
                qe = q + slip
                rows.append((r.game_start_ts, (y - qe) / qe, y, qe))
                break
    return pd.DataFrame(rows, columns=["ts", "pnl", "won", "q"])

print("=== Elo-confirmation sweep (bucket [0.80,0.97], slip=0.01) ===")
for elo_min in [0.0, 0.60, 0.70, 0.75, 0.80]:
    b = run(df, elo_min, 0.01)
    if len(b) < 20:
        continue
    roi, l, h = summ(b["pnl"])
    print(f"elo>={elo_min:.2f}: n={len(b):4d} hit={b['won'].mean():.3f} ROI={roi*100:6.1f}% [{l*100:6.1f},{h*100:6.1f}]  totalPnL={b['pnl'].sum():+.2f}")

print("\n=== FINAL SPEC: bucket [0.80,0.97], elo>=0.70, by slippage ===")
for slip, lab in [(0.0, "maker"), (0.01, "taker 1c"), (0.02, "taker 2c")]:
    b = run(df, 0.70, slip)
    roi, l, h = summ(b["pnl"])
    print(f"{lab:9}: n={len(b)} hit={b['won'].mean():.3f} ROI={roi*100:.1f}% [{l*100:.1f},{h*100:.1f}] totalPnL={b['pnl'].sum():+.2f}")

b = run(df, 0.70, 0.01)
b["year_half"] = b["ts"].dt.year.astype(str) + "H" + ((b["ts"].dt.month > 6) + 1).astype(str)
print("\nby half-year (slip=0.01):")
g = b.groupby("year_half").agg(n=("pnl", "size"), hit=("won", "mean"), pnl=("pnl", "sum"))
g["roi%"] = (g["pnl"] / g["n"] * 100).round(1)
print(g.round(3).to_string())

b["cum"] = b["pnl"].cumsum()
b.to_parquet("data_raw/final_strategy_bets.parquet")
print(f"\nsaved {len(b)} bets; final cumulative PnL = {b['cum'].iloc[-1]:+.2f} units on flat $1 stakes")

# equity curve chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(b["ts"], b["cum"], color="#16a34a", lw=1.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_title(f"Buy PM favorites 80-97c @24h with Elo>=0.70 confirmation - "
             f"{len(b)} bets, PnL {b['cum'].iloc[-1]:+.1f} units (1c slippage)")
ax.set_ylabel("cumulative PnL ($1 flat stakes)")
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("final_strategy_equity.png", dpi=130)
print("saved final_strategy_equity.png")
