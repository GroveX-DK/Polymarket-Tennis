"""Implementable early-market strategy:

Blend calibrated Elo with the PM price 24h before start (both observable at
trade time, no bookmaker data => no contamination). Bet when the blend
diverges from the 24h price. Two exit modes:
  - hold to resolution
  - exit at match start (capture convergence)
Walk-forward: fit blend weights on the first half of the sample, trade the
second half (and report reverse split as sensitivity).
"""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(11)

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[(df["td_comment"] == "Completed") & (df["winner_final_price"] > 0.5)]
df = df.dropna(subset=["p_elo_cal", "p0_24h", "p0_0h"]).copy()
df = df[(df["p0_24h"] > 0.03) & (df["p0_24h"] < 0.97)]
# snapshot must be a reasonably fresh trade (within 12h of the 24h cutoff)
df["age24_h"] = ((df["game_start_ts"] - pd.Timestamp(0, tz="utc")).dt.total_seconds()
                 - 24 * 3600 - df["p0_24h_t"]) / 3600
df = df[df["age24_h"] <= 12]
df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])
df = df.sort_values("game_start_ts").reset_index(drop=True)
print("sample:", len(df), "| span:", df["game_start_ts"].min().date(), "->", df["game_start_ts"].max().date())

def logit(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def sigmoid(z): return 1 / (1 + np.exp(-z))
def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit_logistic(X, y, iters=3000, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        w -= lr * (X.T @ (p - y)) / len(y)
    return w

def summarize(pnl, stake):
    tot = stake.sum()
    roi = pnl.sum() / tot
    idxs = rng.integers(0, len(pnl), (4000, len(pnl)))
    rois = pnl[idxs].sum(1) / np.maximum(stake[idxs].sum(1), 1e-9)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return roi, lo, hi

def run_split(train, test, label):
    Xtr = np.column_stack([logit(train["p0_24h"]), logit(train["p_elo0"]), np.ones(len(train))])
    w = fit_logistic(Xtr, train["y0"].to_numpy())
    Xte = np.column_stack([logit(test["p0_24h"]), logit(test["p_elo0"]), np.ones(len(test))])
    p_blend = sigmoid(Xte @ w)
    print(f"\n--- {label}: weights pm24={w[0]:.3f} elo={w[1]:.3f} const={w[2]:.3f} ---")
    print(f"  test logloss: blend={logloss(p_blend, test['y0']):.4f}  pm24={logloss(test['p0_24h'], test['y0']):.4f}")
    t = test.copy(); t["p_blend0"] = p_blend

    for slip in [0.01, 0.02]:
        for thr in [0.02, 0.03, 0.05]:
            # hold to resolution
            pnl_h, stake_h, won = [], [], []
            # exit at start
            pnl_x, stake_x = [], []
            for r in t.itertuples():
                for (pb, q24, q0h, y) in [(r.p_blend0, r.p0_24h, r.p0_0h, r.y0),
                                          (1 - r.p_blend0, 1 - r.p0_24h, 1 - r.p0_0h, 1 - r.y0)]:
                    qe = q24 + slip
                    if pb - qe > thr and 0 < qe < 1:
                        pnl_h.append((y - qe) / qe); stake_h.append(1.0); won.append(y)
                        pnl_x.append(((q0h - slip) - qe) / qe); stake_x.append(1.0)
                        break
            if len(pnl_h) < 10:
                continue
            pnl_h, stake_h = np.array(pnl_h), np.array(stake_h)
            pnl_x, stake_x = np.array(pnl_x), np.array(stake_x)
            r1, lo1, hi1 = summarize(pnl_h, stake_h)
            r2, lo2, hi2 = summarize(pnl_x, stake_x)
            print(f"  slip={slip:.2f} thr={thr:.2f}: n={len(pnl_h):4d} hit={np.mean(won):.3f} | "
                  f"hold ROI={r1*100:6.1f}% [{lo1*100:6.1f},{hi1*100:6.1f}] | "
                  f"exit@start ROI={r2*100:6.1f}% [{lo2*100:6.1f},{hi2*100:6.1f}]")
    return t

n = len(df)
half = df["game_start_ts"].quantile(0.5)
early, late = df[df["game_start_ts"] <= half], df[df["game_start_ts"] > half]
run_split(early, late, f"train early half (n={len(early)}) -> trade late half (n={len(late)})")
run_split(late, early, f"REVERSE sanity: train late -> trade early")

# fixed simple blend (no fitting at all): 80/20 pm/elo in logit space
print("\n--- No-fit blend 0.8*logit(pm24)+0.2*logit(elo), full sample ---")
p_blend = sigmoid(0.8 * logit(df["p0_24h"]) + 0.2 * logit(df["p_elo0"]))
t = df.copy(); t["p_blend0"] = p_blend
for slip in [0.01, 0.02]:
    for thr in [0.02, 0.03, 0.05]:
        pnl_h, stake_h, won, pnl_x = [], [], [], []
        for r in t.itertuples():
            for (pb, q24, q0h, y) in [(r.p_blend0, r.p0_24h, r.p0_0h, r.y0),
                                      (1 - r.p_blend0, 1 - r.p0_24h, 1 - r.p0_0h, 1 - r.y0)]:
                qe = q24 + slip
                if pb - qe > thr and 0 < qe < 1:
                    pnl_h.append((y - qe) / qe); stake_h.append(1.0); won.append(y)
                    pnl_x.append(((q0h - slip) - qe) / qe)
                    break
        if len(pnl_h) < 10:
            continue
        pnl_h, stake_h, pnl_x = np.array(pnl_h), np.array(stake_h), np.array(pnl_x)
        r1, lo1, hi1 = summarize(pnl_h, stake_h)
        r2, lo2, hi2 = summarize(pnl_x, stake_h)
        print(f"  slip={slip:.2f} thr={thr:.2f}: n={len(pnl_h):4d} hit={np.mean(won):.3f} | "
              f"hold ROI={r1*100:6.1f}% [{lo1*100:6.1f},{hi1*100:6.1f}] | "
              f"exit@start ROI={r2*100:6.1f}% [{lo2*100:6.1f},{hi2*100:6.1f}]")
