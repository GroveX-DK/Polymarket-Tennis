"""Summary charts for the report."""
import json, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[(df["td_comment"] == "Completed") & (df["winner_final_price"] > 0.5)]
df = df.dropna(subset=["AvgW", "AvgL", "p_elo_cal"]).copy()
df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# --- 1: PM calibration at 1h ---
ax = axes[0]
sub = df.dropna(subset=["p0_1h"]); sub = sub[(sub["p0_1h"] > 0.02) & (sub["p0_1h"] < 0.98)]
both = pd.concat([pd.DataFrame({"p": sub["p0_1h"], "y": sub["y0"]}),
                  pd.DataFrame({"p": 1 - sub["p0_1h"], "y": 1 - sub["y0"]})])
cb = both.groupby(pd.cut(both["p"], np.arange(0, 1.05, .1)), observed=True).agg(
    pred=("p", "mean"), actual=("y", "mean"), n=("p", "size"))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
ax.plot(cb["pred"], cb["actual"], "o-", color="#2563eb", label="PM price 1h before start")
ax.set_xlabel("Polymarket price"); ax.set_ylabel("actual win rate")
ax.set_title(f"Polymarket ATP prices are well calibrated (n={len(sub)})")
ax.legend(); ax.grid(alpha=.3)

# --- 2: log-loss comparison ---
ax = axes[1]
def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
labels, vals = [], []
for snap, lab in [("p0_24h", "PM 24h out"), ("p0_6h", "PM 6h out"), ("p0_1h", "PM 1h out"), ("p0_0h", "PM at start")]:
    s = df.dropna(subset=[snap]); s = s[(s[snap] > .02) & (s[snap] < .98)]
    labels.append(lab); vals.append(logloss(s[snap], s["y0"]))
s = df.dropna(subset=["p0_0h"]); s = s[(s["p0_0h"] > .02) & (s["p0_0h"] < .98)]
avg_w = (1 / s["AvgW"]) / (1 / s["AvgW"] + 1 / s["AvgL"])
p_avg0 = np.where(s["winner_is_outcome0"], avg_w, 1 - avg_w)
labels += ["Bookmaker avg*", "Elo (this model)"]
vals += [logloss(p_avg0, s["y0"]), logloss(s["p_elo0"], s["y0"])]
colors = ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#f59e0b", "#ef4444"]
ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
ax.set_xlim(0.55, 0.62)
ax.set_xlabel("log-loss (lower = sharper)")
ax.set_title("Prediction quality on matched ATP matches")
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
ax.grid(alpha=.3, axis="x")

# --- 3: strategy ROI with CI ---
ax = axes[2]
rows = [
    ("Elo edge>3% @1h", -5.7, -13.1, 1.7),
    ("Elo edge>8% @1h", -6.5, -17.6, 5.2),
    ("Books vs PM @1h (capped)", -3.0, -64.9, 89.6),
    ("Books vs PM @24h (capped)*", 8.6, -8.3, 25.7),
    ("Buy favorites @24h", -0.9, -5.1, 3.4),
    ("Buy longshots @24h", -7.1, -18.4, 4.3),
    ("Momentum @1h", -5.9, -14.2, 2.6),
    ("Fade the move @1h", 3.1, -11.5, 18.1),
]
ys = np.arange(len(rows))[::-1]
for (lab, roi, lo, hi), y in zip(rows, ys):
    color = "#16a34a" if lo > 0 else ("#ef4444" if hi < 0 else "#6b7280")
    ax.plot([lo, hi], [y, y], "-", color=color, lw=2)
    ax.plot(roi, y, "o", color=color, ms=7)
ax.axvline(0, color="k", lw=1)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlabel("ROI % (dot) with 95% bootstrap CI")
ax.set_title("All strategies: no CI clears zero")
ax.grid(alpha=.3, axis="x")

plt.tight_layout()
plt.savefig("report_charts.png", dpi=130, bbox_inches="tight")
print("saved report_charts.png")
