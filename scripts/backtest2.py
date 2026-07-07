"""Honest backtest v2.

Fixes look-ahead contamination found in tennis-data 2025/26 odds
(some rows carry post-result odds). Three strategy families:
  A) Elo-only (clean)
  B) Book-avg vs PM, capped divergence (partially clean, caveated)
  C) PM-internal biases (fully clean): early-price calibration & fav bias
"""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(7)

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[df["td_comment"] == "Completed"]
df = df[df["winner_final_price"] > 0.5]
df = df.dropna(subset=["AvgW", "AvgL", "p_elo_cal"]).copy()

df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])
avg_w = (1 / df["AvgW"]) / (1 / df["AvgW"] + 1 / df["AvgL"])
df["p_avg0"] = np.where(df["winner_is_outcome0"], avg_w, 1 - avg_w)
df["year"] = df["game_start_ts"].dt.year
print("clean:", len(df))

def summarize(b):
    tot = b["stake"].sum()
    roi = b["pnl"].sum() / tot
    idxs = rng.integers(0, len(b), (4000, len(b)))
    rois = b["pnl"].to_numpy()[idxs].sum(1) / np.maximum(b["stake"].to_numpy()[idxs].sum(1), 1e-9)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return roi, lo, hi

def run(s, model_col, snap_col, thr, slip, cap=None):
    s = s.dropna(subset=[snap_col, model_col]).copy()
    s = s[(s[snap_col] > 0.02) & (s[snap_col] < 0.98)]
    if cap is not None:
        s = s[(s[model_col] - s[snap_col]).abs() <= cap]
    bets = []
    for r in s.itertuples():
        q0 = getattr(r, snap_col)
        p0 = getattr(r, model_col)
        for (p, q, y) in [(p0, q0, r.y0), (1 - p0, 1 - q0, 1 - r.y0)]:
            qe = q + slip
            if p - qe > thr and 0 < qe < 1:
                bets.append((r.game_start_ts, 1.0, (y - qe) / qe, y, qe))
                break
    if len(bets) < 5:
        return None
    b = pd.DataFrame(bets, columns=["ts", "stake", "pnl", "won", "q"])
    roi, lo, hi = summarize(b)
    return {"n": len(b), "hit": b["won"].mean(), "roi": roi, "lo": lo, "hi": hi, "bets": b}

# ---------- B: capped book-avg strategy ----------
print("\n=== B: Book-avg vs PM, divergence capped at 0.15 (look-ahead mitigation) ===")
print(f"{'snap':7} {'thr':>5} {'slip':>5} {'n':>5} {'hit':>6} {'ROI%':>7} {'95% CI':>16}")
for snap in ["p0_24h", "p0_1h", "p0_0h"]:
    for thr in [0.02, 0.03, 0.05]:
        r = run(df, "p_avg0", snap, thr, 0.01, cap=0.15)
        if r:
            print(f"{snap:7} {thr:5.2f} {0.01:5.2f} {r['n']:5d} {r['hit']:6.3f} {r['roi']*100:7.1f} [{r['lo']*100:6.1f},{r['hi']*100:6.1f}]")

# per-year for the 1h spec
print("\nper-year (1h, thr=0.03, slip=0.01, cap=0.15):")
for y in [2025, 2026]:
    r = run(df[df["year"] == y], "p_avg0", "p0_1h", 0.03, 0.01, cap=0.15)
    if r:
        print(f"  {y}: n={r['n']} hit={r['hit']:.3f} ROI={r['roi']*100:.1f}% [{r['lo']*100:.1f},{r['hi']*100:.1f}]")

# ---------- C: PM-internal ----------
print("\n=== C1: calibration of PM early price (24h) ===")
sub = df.dropna(subset=["p0_24h"])
sub = sub[(sub["p0_24h"] > 0.02) & (sub["p0_24h"] < 0.98)]
both = pd.concat([pd.DataFrame({"p": sub["p0_24h"], "y": sub["y0"]}),
                  pd.DataFrame({"p": 1 - sub["p0_24h"], "y": 1 - sub["y0"]})])
cb = both.groupby(pd.cut(both["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]), observed=True).agg(
    pred=("p", "mean"), actual=("y", "mean"), n=("p", "size"))
print(cb.to_string())

print("\n=== C2: buy PM favorite / longshot at 24h, hold to resolution (slip=0.01) ===")
for side, lo_, hi_ in [("favorite", 0.60, 0.95), ("longshot", 0.05, 0.40), ("mid", 0.40, 0.60)]:
    bets = []
    for r in sub.itertuples():
        for (q, y) in [(r.p0_24h, r.y0), (1 - r.p0_24h, 1 - r.y0)]:
            if lo_ <= q <= hi_:
                qe = q + 0.01
                bets.append((1.0, (y - qe) / qe, y))
                if side != "mid":
                    break
    b = pd.DataFrame(bets, columns=["stake", "pnl", "won"])
    if side == "mid":
        b = b.iloc[::2]  # one side only to avoid double-count
    roi, lo, hi = summarize(b)
    print(f"  {side:9} n={len(b):4d} hit={b['won'].mean():.3f} ROI={roi*100:6.1f}% [{lo*100:6.1f},{hi*100:6.1f}]")

print("\n=== C3: momentum 24h->1h, enter at 1h in move direction if |move|>=x (slip=0.01) ===")
sub2 = df.dropna(subset=["p0_24h", "p0_1h"])
sub2 = sub2[(sub2["p0_1h"] > 0.02) & (sub2["p0_1h"] < 0.98)]
for x in [0.03, 0.05, 0.08]:
    bets = []
    for r in sub2.itertuples():
        mv = r.p0_1h - r.p0_24h
        if abs(mv) < x:
            continue
        if mv > 0:
            q, y = r.p0_1h, r.y0
        else:
            q, y = 1 - r.p0_1h, 1 - r.y0
        qe = q + 0.01
        bets.append((1.0, (y - qe) / qe, y))
    b = pd.DataFrame(bets, columns=["stake", "pnl", "won"])
    if len(b) > 5:
        roi, lo, hi = summarize(b)
        print(f"  |move|>={x:.2f}: n={len(b):4d} hit={b['won'].mean():.3f} ROI={roi*100:6.1f}% [{lo*100:6.1f},{hi*100:6.1f}]")

print("\n=== C4: contrarian 24h->1h (fade the move) ===")
for x in [0.03, 0.05, 0.08]:
    bets = []
    for r in sub2.itertuples():
        mv = r.p0_1h - r.p0_24h
        if abs(mv) < x:
            continue
        if mv > 0:
            q, y = 1 - r.p0_1h, 1 - r.y0
        else:
            q, y = r.p0_1h, r.y0
        qe = q + 0.01
        bets.append((1.0, (y - qe) / qe, y))
    b = pd.DataFrame(bets, columns=["stake", "pnl", "won"])
    if len(b) > 5:
        roi, lo, hi = summarize(b)
        print(f"  |move|>={x:.2f}: n={len(b):4d} hit={b['won'].mean():.3f} ROI={roi*100:6.1f}% [{lo*100:6.1f},{hi*100:6.1f}]")

# ---------- A: Elo per-year recap ----------
print("\n=== A: Elo-only recap (1h, slip=0.01) ===")
for thr in [0.03, 0.08]:
    r = run(df, "p_elo0", "p0_1h", thr, 0.01)
    if r:
        print(f"  thr={thr:.2f}: n={r['n']} hit={r['hit']:.3f} ROI={r['roi']*100:.1f}% [{r['lo']*100:.1f},{r['hi']*100:.1f}]")
