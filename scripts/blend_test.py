"""Does Elo (or book-avg) add information beyond the PM price itself?
Fit logistic y ~ a*logit(pm) + b*logit(other) + c on 2025, evaluate on 2026.
"""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[(df["td_comment"] == "Completed") & (df["winner_final_price"] > 0.5)]
df = df.dropna(subset=["AvgW", "AvgL", "p_elo_cal", "p0_1h"]).copy()
df = df[(df["p0_1h"] > 0.02) & (df["p0_1h"] < 0.98)]
df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])
avg_w = (1 / df["AvgW"]) / (1 / df["AvgW"] + 1 / df["AvgL"])
df["p_avg0"] = np.where(df["winner_is_outcome0"], avg_w, 1 - avg_w)
df["year"] = df["game_start_ts"].dt.year

def logit(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def sigmoid(z): return 1 / (1 + np.exp(-z))
def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit_logistic(X, y, iters=500, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y)
        w -= lr * g
    return w

tr, te = df[df["year"] == 2025], df[df["year"] == 2026]
print(f"train n={len(tr)} test n={len(te)}")

for label, other in [("PM only", None), ("PM + Elo", "p_elo0"), ("PM + BookAvg (caveat)", "p_avg0")]:
    cols = [logit(tr["p0_1h"])]
    cols_te = [logit(te["p0_1h"])]
    if other:
        cols.append(logit(tr[other])); cols_te.append(logit(te[other]))
    Xtr = np.column_stack(cols + [np.ones(len(tr))])
    Xte = np.column_stack(cols_te + [np.ones(len(te))])
    w = fit_logistic(Xtr, tr["y0"].to_numpy())
    p_te = sigmoid(Xte @ w)
    print(f"{label:22} weights={np.round(w, 3)}  test logloss={logloss(p_te, te['y0'].to_numpy()):.4f}  "
          f"(raw PM test logloss={logloss(te['p0_1h'], te['y0']):.4f})")
