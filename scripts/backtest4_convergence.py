"""Convergence strategy test: when the Elo model disagrees with the PM price
24h out, does the price move TOWARD the model by match start?

Trade: buy the model-underpriced side at 24h (+slip), sell at 1h/0h (-slip).
PnL comes from price movement only, never from match resolution.

Artifact controls: volume >= $1k, fresh trades at both snapshots, exclude
placeholder prices (0.48-0.52 at 24h), n_points >= 20, both time halves.
"""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(5)

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[(df["td_comment"] == "Completed") & (df["winner_final_price"] > 0.5)]
df = df.dropna(subset=["p_elo_cal", "p0_24h", "p0_1h", "p0_0h"]).copy()
df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])

# --- artifact filters ---
df = df[df["volume"].fillna(0) >= 1000]
df = df[df["n_points"] >= 20]
t0 = df["game_start_ts"].astype("int64") / 1e9
df = df[(t0 - 24 * 3600 - df["p0_24h_t"]) / 3600 <= 6]   # fresh 24h trade
df = df[(t0 - df["p0_0h_t"]) / 3600 <= 2]                 # fresh 0h trade
df = df[(df["p0_24h"] < 0.48) | (df["p0_24h"] > 0.52)]    # kill placeholder prints
df = df[(df["p0_24h"] > 0.05) & (df["p0_24h"] < 0.95)]
df = df.sort_values("game_start_ts").reset_index(drop=True)
print(f"sample after filters: {len(df)}")

# --- 1. does the price move toward the model? ---
gap = (df["p_elo0"] - df["p0_24h"]).to_numpy()
move = (df["p0_0h"] - df["p0_24h"]).to_numpy()
slope, intercept = np.polyfit(gap, move, 1)
resid = move - (slope * gap + intercept)
se = np.sqrt(np.sum(resid ** 2) / (len(gap) - 2) / np.sum((gap - gap.mean()) ** 2))
print(f"\nregression: (0h price - 24h price) = {intercept:+.4f} + {slope:.3f} * (elo - 24h price)")
print(f"slope t-stat = {slope / se:.2f}   (positive & significant => prices converge toward Elo)")
corr = np.corrcoef(gap, move)[0, 1]
print(f"correlation = {corr:.3f}")

# also: does the price move toward the RESULT (sanity: it should)
gap_y = (df["y0"] - df["p0_24h"]).to_numpy()
s2 = np.polyfit(gap_y, move, 1)[0]
print(f"sanity: slope of move on (outcome - 24h price) = {s2:.3f} (should be clearly > 0)")

# --- 2. trading simulation ---
def summ(pnl):
    idxs = rng.integers(0, len(pnl), (4000, len(pnl)))
    rois = pnl.to_numpy()[idxs].mean(1)
    return pnl.mean(), *np.percentile(rois, [2.5, 97.5])

def sim(t, thr, slip, exit_col):
    rows = []
    for r in t.itertuples():
        g = r.p_elo0 - r.p0_24h
        if abs(g) < thr:
            continue
        if g > 0:   # model says outcome0 underpriced -> buy outcome0
            entry = r.p0_24h + slip
            exitp = getattr(r, exit_col) - slip
        else:       # buy outcome1
            entry = (1 - r.p0_24h) + slip
            exitp = (1 - getattr(r, exit_col)) - slip
        if not (0 < entry < 1):
            continue
        rows.append({"ts": r.game_start_ts, "pnl": (exitp - entry) / entry})
    return pd.DataFrame(rows)

half = df["game_start_ts"].quantile(0.5)
print(f"\n{'exit':5} {'thr':>5} {'slip':>5} {'period':>6} | {'n':>4} {'ROI%':>7} {'95% CI':>16}")
for exit_col in ["p0_1h", "p0_0h"]:
    for thr in [0.03, 0.05, 0.10]:
        for slip in [0.01]:
            for label, t in [("all", df), ("early", df[df["game_start_ts"] <= half]),
                             ("late", df[df["game_start_ts"] > half])]:
                b = sim(t, thr, slip, exit_col)
                if len(b) < 15:
                    continue
                roi, lo, hi = summ(b["pnl"])
                print(f"{exit_col:5} {thr:5.2f} {slip:5.2f} {label:>6} | {len(b):4d} {roi*100:7.1f} [{lo*100:6.1f},{hi*100:6.1f}]")

# --- 3. the reverse: fade the model (market wins the argument?) ---
print("\nreverse test (buy the side the model thinks is OVERpriced):")
def sim_rev(t, thr, slip, exit_col):
    rows = []
    for r in t.itertuples():
        g = r.p_elo0 - r.p0_24h
        if abs(g) < thr:
            continue
        if g > 0:
            entry = (1 - r.p0_24h) + slip
            exitp = (1 - getattr(r, exit_col)) - slip
        else:
            entry = r.p0_24h + slip
            exitp = getattr(r, exit_col) - slip
        if not (0 < entry < 1):
            continue
        rows.append({"pnl": (exitp - entry) / entry})
    return pd.DataFrame(rows)
for thr in [0.05, 0.10]:
    b = sim_rev(df, thr, 0.01, "p0_0h")
    if len(b) >= 15:
        roi, lo, hi = summ(b["pnl"])
        print(f"  thr={thr:.2f} exit 0h: n={len(b)} ROI={roi*100:.1f}% [{lo*100:.1f},{hi*100:.1f}]")
