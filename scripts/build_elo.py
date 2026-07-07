"""Build chronological Elo ratings (overall + surface-blended) over tennis-data
2000-2026 and validate predictive quality vs Pinnacle on a holdout period.

Outputs data_raw/elo_predictions.parquet with one row per match:
  pre-match Elo probs for the eventual winner, plus odds columns.
"""
import io, sys
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------- load ----------
hist = pd.read_parquet("data_raw/tennisdata_all.parquet")
cur = pd.read_excel("data_raw/2026.xlsx")
cur["season"] = 2026
cols = [c for c in hist.columns if c in cur.columns]
df = pd.concat([hist[cols], cur[cols]], ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"])
for c in ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL",
          "WRank", "LRank", "WPts", "LPts", "Wsets", "Lsets", "Best of"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# order within a day by round so tournament progression is respected
round_order = {"1st Round": 0, "2nd Round": 1, "3rd Round": 2, "4th Round": 3,
               "Round Robin": 2, "Quarterfinals": 4, "Semifinals": 5, "The Final": 6}
df["rnd"] = df["Round"].map(round_order).fillna(1)
df = df.sort_values(["Date", "rnd"], kind="stable").reset_index(drop=True)
df = df[df["Comment"].isin(["Completed", "Retired", "Awarded", "retired", "Sched"]) | True]  # keep all; walkovers still count as results in tennis-data
print("matches:", len(df), "| span:", df["Date"].min().date(), "->", df["Date"].max().date())

# ---------- Elo ----------
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
elo = {}        # name -> overall rating
elo_s = {}      # (name, surface) -> rating
n_played = {}   # name -> match count
n_played_s = {}

def get_elo(p): return elo.get(p, 1500.0)
def get_elo_s(p, s): return elo_s.get((p, s), 1500.0)

def k_factor(n): return 250.0 / ((n + 5) ** 0.4)

rows = []
W = df["Winner"].to_numpy(); L = df["Loser"].to_numpy()
S = df["Surface"].fillna("Hard").to_numpy()
BO = df["Best of"].fillna(3).to_numpy()

for i in range(len(df)):
    w, l, s = W[i], L[i], S[i]
    ew, el = get_elo(w), get_elo(l)
    ews, els = get_elo_s(w, s), get_elo_s(l, s)
    # blended pre-match prob that the (eventual) winner wins
    d_all = ew - el
    d_srf = ews - els
    d_blend = 0.6 * d_all + 0.4 * d_srf
    p_w = 1.0 / (1.0 + 10 ** (-d_blend / 400.0))
    p_w_all = 1.0 / (1.0 + 10 ** (-d_all / 400.0))
    rows.append((p_w, p_w_all, ew, el, n_played.get(w, 0), n_played.get(l, 0)))
    # updates
    nw, nl = n_played.get(w, 0), n_played.get(l, 0)
    kw, kl = k_factor(nw), k_factor(nl)
    exp_w = 1.0 / (1.0 + 10 ** (-(ew - el) / 400.0))
    elo[w] = ew + kw * (1 - exp_w)
    elo[l] = el - kl * (1 - exp_w)
    nws, nls = n_played_s.get((w, s), 0), n_played_s.get((l, s), 0)
    kws, kls = k_factor(nws), k_factor(nls)
    exp_ws = 1.0 / (1.0 + 10 ** (-(ews - els) / 400.0))
    elo_s[(w, s)] = ews + kws * (1 - exp_ws)
    elo_s[(l, s)] = els - kls * (1 - exp_ws)
    n_played[w] = nw + 1; n_played[l] = nl + 1
    n_played_s[(w, s)] = nws + 1; n_played_s[(l, s)] = nls + 1

pred = pd.DataFrame(rows, columns=["p_elo", "p_elo_all", "elo_w", "elo_l", "n_w", "n_l"])
out = pd.concat([df.reset_index(drop=True), pred], axis=1)

# ---------- calibration: fit shrink factor s on 2005-2020 ----------
# p = sigma(s * logit-equivalent elo diff); overconfident elo => s < 1
train = out[(out["season"] >= 2005) & (out["season"] <= 2020)]
train = train[(train["n_w"] >= 10) & (train["n_l"] >= 10)]
d = np.log10(train["p_elo"] / (1 - train["p_elo"]))  # elo diff in dex units

def ll_for_scale(s):
    p = 1.0 / (1.0 + 10 ** (-s * d))
    return -np.mean(np.log(np.clip(p, 1e-6, 1)))

grid = np.arange(0.5, 1.21, 0.025)
lls = [ll_for_scale(s) for s in grid]
s_best = float(grid[int(np.argmin(lls))])
print(f"\ncalibration scale s = {s_best:.3f} (train logloss {min(lls):.4f} vs raw {ll_for_scale(1.0):.4f})")

d_all_matches = np.log10(out["p_elo"] / (1 - out["p_elo"]))
out["p_elo_cal"] = 1.0 / (1.0 + 10 ** (-s_best * d_all_matches))
out.to_parquet("data_raw/elo_predictions.parquet")

# ---------- validation vs Pinnacle (vig-free) on 2021-2024 ----------
val = out[(out["season"] >= 2021) & (out["season"] <= 2024)].copy()
val = val.dropna(subset=["PSW", "PSL"])
val = val[(val["n_w"] >= 10) & (val["n_l"] >= 10)]  # both players warmed up
pw_ps = (1 / val["PSW"]) / (1 / val["PSW"] + 1 / val["PSL"])

def logloss(p): return -np.mean(np.log(np.clip(p, 1e-6, 1 - 1e-6)))
def acc(p): return np.mean(p > 0.5)

print(f"\nValidation 2021-2024 (n={len(val)}):")
print(f"  Elo blended : logloss={logloss(val['p_elo']):.4f}  acc={acc(val['p_elo']):.3f}")
print(f"  Elo calib.  : logloss={logloss(val['p_elo_cal']):.4f}  acc={acc(val['p_elo_cal']):.3f}")
print(f"  Elo overall : logloss={logloss(val['p_elo_all']):.4f}  acc={acc(val['p_elo_all']):.3f}")
print(f"  Pinnacle    : logloss={logloss(pw_ps):.4f}  acc={acc(pw_ps):.3f}")

# calibration curve (both perspectives) for calibrated elo
both = pd.concat([
    pd.DataFrame({"p": val["p_elo_cal"], "y": 1}),
    pd.DataFrame({"p": 1 - val["p_elo_cal"], "y": 0}),
])
cb = both.groupby(pd.cut(both["p"], np.arange(0, 1.05, 0.1)), observed=True).agg(
    pred=("p", "mean"), actual=("y", "mean"), n=("p", "size"))
print("\nCalibration (calibrated Elo):")
print(cb.to_string())
