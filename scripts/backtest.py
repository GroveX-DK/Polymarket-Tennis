"""Backtest strategies vs historical Polymarket ATP match prices.

Buy outcome i at snapshot price when model prob exceeds price by threshold.
Flat $1 cost per bet. PnL = (1{win} - q) / q per $1.
"""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(7)

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
snaps = snaps[~snaps.get("error").notna()] if "error" in snaps.columns else snaps
df = md.merge(snaps, on="market_id", how="inner")
print("merged:", len(df))

# ---------- clean ----------
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[df["td_comment"] == "Completed"]          # avoid retirement resolution quirks
df = df[df["winner_final_price"] > 0.5]           # PM resolution agrees with tennis-data
df = df.dropna(subset=["AvgW", "AvgL", "p_elo_cal"])
print("clean:", len(df))

# orient everything to outcome0
df["y0"] = df["winner_is_outcome0"].astype(float)          # 1 if outcome0 won
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])
ps_w = (1 / df["PSW"]) / (1 / df["PSW"] + 1 / df["PSL"])   # vig-free pinnacle prob of td winner
df["p_ps0"] = np.where(df["winner_is_outcome0"], ps_w, 1 - ps_w)
avg_w = (1 / df["AvgW"]) / (1 / df["AvgW"] + 1 / df["AvgL"])
df["p_avg0"] = np.where(df["winner_is_outcome0"], avg_w, 1 - avg_w)

def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# ---------- market quality: whose probs are best on this sample? ----------
for snap in ["p0_24h", "p0_6h", "p0_1h", "p0_0h"]:
    sub = df.dropna(subset=[snap])
    sub = sub[(sub[snap] > 0.01) & (sub[snap] < 0.99)]
    ps_sub = sub.dropna(subset=["p_ps0"])
    print(f"{snap}: n={len(sub)}  PM logloss={logloss(sub[snap], sub['y0']):.4f}  "
          f"Avg logloss={logloss(sub['p_avg0'], sub['y0']):.4f}  "
          f"Elo logloss={logloss(sub['p_elo0'], sub['y0']):.4f}  "
          f"PS logloss={logloss(ps_sub['p_ps0'], ps_sub['y0']):.4f} (n={len(ps_sub)})")

# ---------- backtest ----------
def run(sub, model_col, snap_col, thr, slip, min_vol=0, kelly=False, verbose=False):
    s = sub.dropna(subset=[snap_col, model_col]).copy()
    s = s[(s[snap_col] > 0.02) & (s[snap_col] < 0.98)]
    if min_vol:
        s = s[s["volume"] >= min_vol]
    q0 = s[snap_col].to_numpy()
    pm_model = s[model_col].to_numpy()
    y = s["y0"].to_numpy()
    bets = []
    for i in range(len(s)):
        for (p_i, q_i, y_i) in [(pm_model[i], q0[i], y[i]),
                                (1 - pm_model[i], 1 - q0[i], 1 - y[i])]:
            q_eff = q_i + slip
            if p_i - q_eff > thr and 0 < q_eff < 1:
                edge = p_i - q_eff
                stake = 1.0
                if kelly:
                    b = (1 - q_eff) / q_eff
                    f = max(0.0, (p_i * b - (1 - p_i)) / b) * 0.25
                    stake = f
                pnl = stake * ((y_i - q_eff) / q_eff)
                bets.append((s.iloc[i]["game_start_ts"], edge, stake, pnl, y_i, q_eff))
                break
    if not bets:
        return None
    b = pd.DataFrame(bets, columns=["ts", "edge", "stake", "pnl", "won", "q"])
    tot_stake = b["stake"].sum()
    roi = b["pnl"].sum() / tot_stake if tot_stake else 0
    # bootstrap CI on ROI
    idxs = rng.integers(0, len(b), (2000, len(b)))
    rois = b["pnl"].to_numpy()[idxs].sum(1) / np.maximum(b["stake"].to_numpy()[idxs].sum(1), 1e-9)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return {"n": len(b), "hit": b["won"].mean(), "avg_q": b["q"].mean(),
            "pnl": b["pnl"].sum(), "roi": roi, "roi_lo": lo, "roi_hi": hi, "bets": b}

print("\n=== PRIMARY: Bookmaker-avg vig-free vs PM @1h, thr=0.03, slip=0.01 ===")
r = run(df, "p_avg0", "p0_1h", 0.03, 0.01)
if r:
    print(f"n={r['n']} hit={r['hit']:.3f} avg_price={r['avg_q']:.3f} "
          f"PnL=${r['pnl']:.1f} ROI={r['roi']*100:.1f}% CI=[{r['roi_lo']*100:.1f}%, {r['roi_hi']*100:.1f}%]")

print("\n=== Sweeps (flat $1, slip=0.01, snapshot 1h) ===")
print(f"{'model':8} {'thr':>5} {'n':>5} {'hit':>6} {'ROI%':>7} {'CI':>18}")
for model in ["p_ps0", "p_avg0", "p_elo0"]:
    for thr in [0.02, 0.03, 0.05, 0.08, 0.12]:
        r = run(df, model, "p0_1h", thr, 0.01)
        if r:
            print(f"{model:8} {thr:5.2f} {r['n']:5d} {r['hit']:6.3f} {r['roi']*100:7.1f} "
                  f"[{r['roi_lo']*100:6.1f},{r['roi_hi']*100:6.1f}]")

print("\n=== Slippage sensitivity (Avg, thr=0.03, 1h) ===")
for slip in [0.0, 0.01, 0.02, 0.03]:
    r = run(df, "p_avg0", "p0_1h", 0.03, slip)
    if r:
        print(f"slip={slip:.2f}: n={r['n']} ROI={r['roi']*100:.1f}% [{r['roi_lo']*100:.1f},{r['roi_hi']*100:.1f}]")

print("\n=== Snapshot timing sensitivity (Avg, thr=0.03, slip=0.01) ===")
for snap in ["p0_24h", "p0_6h", "p0_1h", "p0_0h"]:
    r = run(df, "p_avg0", snap, 0.03, 0.01)
    if r:
        print(f"{snap}: n={r['n']} ROI={r['roi']*100:.1f}% [{r['roi_lo']*100:.1f},{r['roi_hi']*100:.1f}]")

print("\n=== Volume filter (Avg, thr=0.03, slip=0.01, 1h) ===")
for mv in [0, 1000, 5000, 20000]:
    r = run(df, "p_avg0", "p0_1h", 0.03, 0.01, min_vol=mv)
    if r:
        print(f"min_vol={mv}: n={r['n']} ROI={r['roi']*100:.1f}% [{r['roi_lo']*100:.1f},{r['roi_hi']*100:.1f}]")

# blend
df["p_blend0"] = 0.7 * df["p_avg0"] + 0.3 * df["p_elo0"]
print("\n=== Blend 0.7*PS + 0.3*Elo (thr sweep, slip=0.01, 1h) ===")
for thr in [0.02, 0.03, 0.05, 0.08]:
    r = run(df, "p_blend0", "p0_1h", thr, 0.01)
    if r:
        print(f"thr={thr:.2f}: n={r['n']} hit={r['hit']:.3f} ROI={r['roi']*100:.1f}% [{r['roi_lo']*100:.1f},{r['roi_hi']*100:.1f}]")

# save primary bets for reporting
r = run(df, "p_avg0", "p0_1h", 0.03, 0.01)
if r:
    b = r["bets"].sort_values("ts")
    b["cum"] = b["pnl"].cumsum()
    b.to_parquet("data_raw/backtest_bets_primary.parquet")
    monthly = b.set_index("ts")["pnl"].resample("MS").agg(["sum", "count"])
    print("\nMonthly PnL (primary):")
    print(monthly.to_string())
