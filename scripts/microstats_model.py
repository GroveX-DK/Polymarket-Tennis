"""Serve/return micro-stats + context GBM model (items 4 & 5 of the accuracy list).

Features, all computed strictly from matches BEFORE the one being predicted:
  #4 micro-stats (EWMA, halflife 25 matches, overall + per-surface):
     serve points won %, return points won %, ace rate, double-fault rate,
     hold %, break-point save %
  #5 context: best-of-5, round, indoor, home country, retirement/walkover risk
  base: overall + surface Elo (same K-schedule as build_elo), rank, rank points,
        age, height

Model: HistGradientBoostingClassifier, trained on 2000-2020, both match
orientations (symmetric). Eval: 2021-2024 holdout vs calibrated Elo on the
identical rows; 2025 Polymarket matched sample vs actual PM prices.

Data: TML-Database (Sackmann schema) via scripts/fetch_tml.py.
"""
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(7)

ROUND_ORD = {"RR": 0, "ER": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
             "QF": 5, "SF": 6, "BR": 7, "F": 8}
HL_STAT, HL_RET = 25, 60          # EWMA halflives (matches)
A_STAT = 1 - 0.5 ** (1 / HL_STAT)
A_RET = 1 - 0.5 ** (1 / HL_RET)
MIN_N, MIN_STAT = 15, 5           # history required to emit a row

HOME = {  # tourney_name -> host IOC (None/absent = no home feature)
    "'s-Hertogenbosch": "NED", "Acapulco": "MEX", "Adelaide": "AUS",
    "Adelaide-1": "AUS", "Adelaide-2": "AUS", "Almaty": "KAZ", "Antalya": "TUR",
    "Antwerp": "BEL", "Astana": "KAZ", "Athens": "GRE", "Atlanta": "USA",
    "ATP Cup": "AUS", "Atp Cup": "AUS", "United Cup": "AUS", "Auckland": "NZL",
    "Australian Open": "AUS", "Bangkok": "THA", "Banja Luka": "BIH",
    "Barcelona": "ESP", "Basel": "SUI", "Bastad": "SWE", "Beijing": "CHN",
    "Belgrade": "SRB", "Bogota": "COL", "Brisbane": "AUS", "Brussels": "BEL",
    "Bucharest": "ROU", "Budapest": "HUN", "Buenos Aires": "ARG",
    "Cabo San Lucas": "MEX", "Cagliari": "ITA", "Canada Masters": "CAN",
    "Casablanca": "MAR", "Chengdu": "CHN", "Chennai": "IND",
    "Cincinnati Masters": "USA", "Cologne 1": "GER", "Cologne 2": "GER",
    "Copenhagen": "DEN", "Cordoba": "ARG", "Costa Do Sauipe": "BRA",
    "Dallas": "USA", "Delray Beach": "USA", "Doha": "QAT", "Dubai": "UAE",
    "Eastbourne": "GBR", "Estoril": "POR", "Florence": "ITA", "Geneva": "SUI",
    "Gijon": "ESP", "Great Ocean Road Open": "AUS", "Gstaad": "SUI",
    "Halle": "GER", "Hamburg": "GER", "Hangzhou": "CHN", "Hong Kong": "HKG",
    "Houston": "USA", "Indian Wells Masters": "USA", "Indianapolis": "USA",
    "Istanbul": "TUR", "Johannesburg": "RSA", "Kitzbuhel": "AUT",
    "Kuala Lumpur": "MAS", "Long Island": "USA", "Los Angeles": "USA",
    "Los Cabos": "MEX", "Lyon": "FRA", "Madrid Masters": "ESP",
    "Mallorca": "ESP", "Marbella": "ESP", "Marrakech": "MAR",
    "Marseille": "FRA", "Melbourne": "AUS", "Memphis": "USA", "Metz": "FRA",
    "Miami Masters": "USA", "Milan": "ITA", "Monte Carlo Masters": "MON",
    "Montpellier": "FRA", "Moscow": "RUS", "Munich": "GER",
    "Murray River Open": "AUS", "Naples": "ITA", "New Haven": "USA",
    "New York": "USA", "New York Open": "USA", "Newport": "USA", "Nice": "FRA",
    "Nottingham": "GBR", "Nur-Sultan": "KAZ", "Palermo": "ITA",
    "Paris Masters": "FRA", "Paris Olympics": "FRA", "Parma": "ITA",
    "Pune": "IND", "Queen's Club": "GBR", "Quito": "ECU",
    "Rio De Janeiro": "BRA", "Rio de Janeiro": "BRA", "Rio Olympics": "BRA",
    "Roland Garros": "FRA", "Rome Masters": "ITA", "Rotterdam": "NED",
    "Salvador": "BRA", "San Diego": "USA", "San Jose": "USA",
    "Santiago": "CHI", "Sao Paulo": "BRA", "Sardinia": "ITA", "Seoul": "KOR",
    "Serbia": "SRB", "Shanghai": "CHN", "Shanghai Masters": "CHN",
    "Shenzhen": "CHN", "Singapore": "SGP", "Sofia": "BUL",
    "St. Petersburg": "RUS", "Stockholm": "SWE", "Stuttgart": "GER",
    "Sydney": "AUS", "Tel Aviv": "ISR", "Tokyo": "JPN",
    "Tokyo Olympics": "JPN", "Toulouse": "FRA", "Umag": "CRO",
    "US Open": "USA", "Valencia": "ESP", "Vienna": "AUT", "Vina del Mar": "CHI",
    "Warsaw": "POL", "Washington": "USA", "Wimbledon": "GBR",
    "Winston-Salem": "USA", "Winston Salem": "USA", "Zagreb": "CRO",
    "Zhuhai": "CHN", "Amersfoort": "NED", "Amsterdam": "NED",
}

# ---------------- load ----------------
frames = []
for y in range(2000, 2027):
    frames.append(pd.read_csv(f"data_raw/tml/{y}.csv", low_memory=False))
df = pd.concat(frames, ignore_index=True)
df = df[df["tourney_level"] != "D"]                       # no Davis Cup
df = df[~df["tourney_name"].str.contains("Next ?Gen|Laver Cup", case=False,
                                         regex=True, na=False)]
df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
df["rnd"] = df["round"].map(ROUND_ORD).fillna(3).astype(int)
df = df.sort_values(["date", "tourney_id", "rnd", "match_num"],
                    kind="stable").reset_index(drop=True)
df["score"] = df["score"].fillna("")
print(f"TML matches loaded: {len(df)}  ({df['date'].min().date()} .. {df['date'].max().date()})")

# ---------------- chronological feature pass ----------------
class P:  # per-player EWMA state
    __slots__ = ("elo", "elo_s", "n", "n_stat", "stats", "stats_s", "ret")
    def __init__(self):
        self.elo, self.elo_s = 1500.0, {}
        self.n, self.n_stat = 0, 0
        self.stats = {}       # name -> ewma value (overall)
        self.stats_s = {}     # (name, surface) -> ewma value
        self.ret = 0.0

players = {}
def get(p):
    if p not in players:
        players[p] = P()
    return players[p]

def ew(d, k, x, a=A_STAT):
    d[k] = x if k not in d else (1 - a) * d[k] + a * x

STATS = ["spw", "rpw", "ace", "df", "hold", "bpsv"]

def match_stats(r, side):  # side "w" or "l"; returns dict or None
    try:
        svpt = float(r[f"{side}_svpt"])
        if not svpt or svpt != svpt:
            return None
        out = {
            "spw": (float(r[f"{side}_1stWon"]) + float(r[f"{side}_2ndWon"])) / svpt,
            "ace": float(r[f"{side}_ace"]) / svpt,
            "df": float(r[f"{side}_df"]) / svpt,
        }
        sg = float(r[f"{side}_SvGms"])
        bf, bs = float(r[f"{side}_bpFaced"]), float(r[f"{side}_bpSaved"])
        if sg > 0:
            out["hold"] = 1.0 - (bf - bs) / sg
        if bf > 0:
            out["bpsv"] = bs / bf
        return out
    except (ValueError, TypeError):
        return None

def feat_vec(pw, pl, surface, r, w_home, l_home):
    """Antisymmetric diffs (winner-minus-loser) + symmetric context."""
    def sd(p, q, name):
        a = p.stats.get(name), q.stats.get(name)
        s = p.stats_s.get((name, surface)), q.stats_s.get((name, surface))
        d = (a[0] - a[1]) if None not in a else np.nan
        ds = (s[0] - s[1]) if None not in s else np.nan
        return d, ds
    es_w = pw.elo_s.get(surface, 1500.0)
    es_l = pl.elo_s.get(surface, 1500.0)
    f = {"elo_d": pw.elo - pl.elo, "elo_s_d": es_w - es_l}
    for name in STATS:
        f[name + "_d"], f[name + "_s_d"] = sd(pw, pl, name)
    f["ret_d"] = pw.ret - pl.ret
    def fnum(v):
        try:
            v = float(v)
            return v if v == v and v > 0 else np.nan
        except (ValueError, TypeError):
            return np.nan
    f["lrank_d"] = -(np.log(fnum(r["winner_rank"])) - np.log(fnum(r["loser_rank"])))
    f["lpts_d"] = np.log1p(fnum(r["winner_rank_points"])) - np.log1p(fnum(r["loser_rank_points"]))
    f["age_d"] = fnum(r["winner_age"]) - fnum(r["loser_age"])
    f["ht_d"] = fnum(r["winner_ht"]) - fnum(r["loser_ht"])
    f["home_d"] = float(w_home) - float(l_home)
    # symmetric context
    f["bo5"] = 1.0 if r["best_of"] == 5 else 0.0
    f["rnd"] = float(r["rnd"])
    f["indoor"] = 1.0 if r.get("indoor") == "I" else 0.0
    f["clay"] = 1.0 if surface == "Clay" else 0.0
    f["grass"] = 1.0 if surface == "Grass" else 0.0
    f["carpet"] = 1.0 if surface == "Carpet" else 0.0
    return f

def k_elo(n):
    return 250.0 / ((n + 5) ** 0.4)

rows, meta = [], []
for r in df.itertuples(index=False):
    r = r._asdict()
    w, l = r["winner_name"], r["loser_name"]
    surface = r["surface"] if isinstance(r["surface"], str) else "Hard"
    pw, pl = get(w), get(l)
    sc = r["score"].upper()
    walkover = "W/O" in sc or "WEA" in sc or sc.strip() in ("", "DEF")
    retired = "RET" in sc or "ABN" in sc or "ABD" in sc or "DEF" in sc

    if not walkover:
        host = HOME.get(r["tourney_name"])
        w_home = host is not None and r["winner_ioc"] == host
        l_home = host is not None and r["loser_ioc"] == host
        if (pw.n >= MIN_N and pl.n >= MIN_N
                and pw.n_stat >= MIN_STAT and pl.n_stat >= MIN_STAT):
            f = feat_vec(pw, pl, surface, r, w_home, l_home)
            rows.append(f)
            d_blend = 0.6 * (pw.elo - pl.elo) + 0.4 * (
                pw.elo_s.get(surface, 1500.0) - pl.elo_s.get(surface, 1500.0))
            meta.append({
                "date": r["date"], "year": r["date"].year,
                "winner": w, "loser": l, "tourney": r["tourney_name"],
                "p_elo_cal": 1.0 / (1.0 + 10.0 ** (-0.85 * d_blend / 400.0)),
            })

    # ---- state updates (after feature emission) ----
    if walkover:
        pl.ret = (1 - A_RET) * pl.ret + A_RET * 1.0
        continue
    exp = 1.0 / (1.0 + 10.0 ** (-(pw.elo - pl.elo) / 400.0))
    pw.elo += k_elo(pw.n) * (1 - exp)
    pl.elo -= k_elo(pl.n) * (1 - exp)
    ews, els = pw.elo_s.get(surface, 1500.0), pl.elo_s.get(surface, 1500.0)
    exps = 1.0 / (1.0 + 10.0 ** (-(ews - els) / 400.0))
    pw.elo_s[surface] = ews + k_elo(pw.n) * (1 - exps)
    pl.elo_s[surface] = els - k_elo(pl.n) * (1 - exps)
    sw, sl = match_stats(r, "w"), match_stats(r, "l")
    if sw and sl:
        sw["rpw"], sl["rpw"] = 1.0 - sl["spw"], 1.0 - sw["spw"]
        for p, s in ((pw, sw), (pl, sl)):
            for kk, v in s.items():
                ew(p.stats, kk, v)
                ew(p.stats_s, (kk, surface), v)
            p.n_stat += 1
    pw.ret = (1 - A_RET) * pw.ret
    pl.ret = (1 - A_RET) * pl.ret + A_RET * (1.0 if retired else 0.0)
    pw.n += 1
    pl.n += 1

X = pd.DataFrame(rows)
M = pd.DataFrame(meta)
FEATS = list(X.columns)
ANTI = [c for c in FEATS if c.endswith("_d")]  # flip sign when swapping players
print(f"feature rows (matches): {len(X)}, features: {len(FEATS)}")

# both orientations for training; symmetric averaging for prediction
def orientations(Xm):
    X1 = Xm.copy(); X2 = Xm.copy()
    flip = [c for c in ANTI if c in Xm.columns]
    X2[flip] = -X2[flip]
    return X1, X2  # y=1 for X1 (winner as p1), y=0 for X2

def fit(Xtr, cols, seed=0):
    X1, X2 = orientations(Xtr[cols])
    Xall = pd.concat([X1, X2], ignore_index=True)
    yall = np.r_[np.ones(len(X1)), np.zeros(len(X2))]
    m = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.08,
        n_iter_no_change=30, random_state=seed)
    m.fit(Xall, yall)
    return m

def predict_sym(m, Xe, cols):
    X1, X2 = orientations(Xe[cols])
    return (m.predict_proba(X1[cols])[:, 1] + 1 - m.predict_proba(X2[cols])[:, 1]) / 2

tr = M["year"] <= 2020
ho = (M["year"] >= 2021) & (M["year"] <= 2024)
print(f"train matches: {tr.sum()}, holdout 2021-2024: {ho.sum()}")

model = fit(X[tr], FEATS)
p_full = predict_sym(model, X[ho], FEATS)
base_model = fit(X[tr], ["elo_d", "elo_s_d"])
p_base = predict_sym(base_model, X[ho], ["elo_d", "elo_s_d"])
p_elo = M.loc[ho, "p_elo_cal"].to_numpy()
y = np.ones(ho.sum())  # winner as p1

def ll(p):  # log-loss of winner-side probs, with accuracy
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(np.log(p)), np.mean(p > 0.5)

print("\n=== holdout 2021-2024 (identical rows) ===")
for lab, p in [("calibrated Elo (report model)", p_elo),
               ("GBM elo-only (ablation)", p_base),
               ("GBM full (micro-stats + context)", p_full)]:
    L, acc = ll(p)
    print(f"{lab:34} log-loss {L:.4f}  accuracy {acc:.3f}")

# bootstrap CI on the log-loss difference (full GBM vs calibrated Elo)
idx = rng.integers(0, ho.sum(), (2000, ho.sum()))
d = (-np.log(np.clip(p_elo, 1e-6, 1)) + np.log(np.clip(p_full, 1e-6, 1)))
diffs = d[idx].mean(1)
print(f"GBM improvement over Elo: {d.mean():+.4f} "
      f"[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}] "
      f"(positive = GBM better)")

imp = permutation_importance(
    model, *[pd.concat(orientations(X[ho].iloc[:2500][FEATS]), ignore_index=True),
             np.r_[np.ones(2500), np.zeros(2500)]],
    n_repeats=3, random_state=0, scoring="neg_log_loss")
order = np.argsort(-imp.importances_mean)
print("\ntop feature importances (permutation, holdout):")
for i in order[:12]:
    print(f"  {FEATS[i]:10} {imp.importances_mean[i]:.4f}")

# ---------------- Polymarket matched-sample eval (2025 portion) ----------------
def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", s.lower())

md = pd.read_parquet("data_raw/matched.parquet")
md = md[md["td_comment"] == "Completed"].copy()
md["td_date"] = pd.to_datetime(md["td_date"])
snaps = pd.DataFrame([json.loads(x) for x in
                      open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
md = md.merge(snaps, on="market_id", how="left")

td_players = set(md["td_winner"]) | set(md["td_loser"])
def parse_td(name):
    toks = name.strip().split(); inis = []
    while toks and re.fullmatch(r"([A-Z]\.)+[A-Z]?\.?", toks[-1]):
        inis.insert(0, toks.pop())
    return norm("".join(toks)), norm("".join(inis))[:1]
sur = {}
for p in td_players:
    s, i = parse_td(p)
    sur.setdefault(s, []).append((p, i))
def to_td(full):
    nf = norm(full); ini = norm(full.split()[0])[:1]
    cands = [(len(s), p) for s, lst in sur.items() if s and nf.endswith(s)
             for p, i in lst if not i or i == ini]
    return max(cands)[1] if cands else None

ev = M[M["year"] >= 2025].copy()
ev["p_gbm"] = predict_sym(model, X[M["year"] >= 2025], FEATS)
ev["wk"] = ev["winner"].map(to_td)
ev["lk"] = ev["loser"].map(to_td)
ev = ev.dropna(subset=["wk", "lk"])
md["key"] = list(zip(md["td_winner"], md["td_loser"]))
ev["key"] = list(zip(ev["wk"], ev["lk"]))
j = ev.merge(md, on="key", suffixes=("", "_md"))
j = j[(j["td_date"] >= j["date"] - pd.Timedelta(days=2))
      & (j["td_date"] <= j["date"] + pd.Timedelta(days=30))]
j["dgap"] = (j["td_date"] - j["date"]).abs().dt.days
j = j.sort_values("dgap").drop_duplicates(subset=["market_id"]).drop_duplicates(subset=["key", "date"])
print(f"\n=== Polymarket matched sample joined: {len(j)} markets (2025 portion) ===")
for col, lab in [("p0_0h", "PM at start"), ("p0_1h", "PM 1h out"), ("p0_24h", "PM 24h out")]:
    if col not in j.columns:
        continue
    sub = j.dropna(subset=[col])
    sub = sub[(sub[col] > 0.01) & (sub[col] < 0.99)]
    pm_pw = np.where(sub["winner_is_outcome0"], sub[col], 1 - sub[col])
    Lpm, accpm = ll(pm_pw)
    Lg, accg = ll(sub["p_gbm"].to_numpy())
    Le, acce = ll(sub["p_elo_cal"].to_numpy())
    print(f"{lab:12} n={len(sub):4d} | PM {Lpm:.4f}/{accpm:.3f} | "
          f"GBM {Lg:.4f}/{accg:.3f} | Elo {Le:.4f}/{acce:.3f}")

# does the GBM add information on top of the PM price? (chronological 2-fold)
from sklearn.linear_model import LogisticRegression
sub = j.dropna(subset=["p0_0h"])
sub = sub[(sub["p0_0h"] > 0.01) & (sub["p0_0h"] < 0.99)].sort_values("td_date")
pm_pw = np.where(sub["winner_is_outcome0"], sub["p0_0h"], 1 - sub["p0_0h"])
lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
Z = np.c_[lo(pm_pw), lo(sub["p_gbm"].to_numpy())]
yy = np.ones(len(Z))
# symmetric training set (both orientations), split chronologically in half
half = len(Z) // 2
print(f"\n=== does GBM add info beyond the PM price at start? (n={len(Z)}) ===")
for name, tr_idx, te_idx in [("fit 1st half -> test 2nd", slice(0, half), slice(half, None)),
                             ("fit 2nd half -> test 1st", slice(half, None), slice(0, half))]:
    Ztr = np.r_[Z[tr_idx], -Z[tr_idx]]
    ytr = np.r_[np.ones(len(Z[tr_idx])), np.zeros(len(Z[tr_idx]))]
    for cols, lab in [([0], "PM only"), ([0, 1], "PM + GBM")]:
        lr = LogisticRegression(C=100.0, fit_intercept=False).fit(Ztr[:, cols], ytr)
        pte = lr.predict_proba(Z[te_idx][:, cols])[:, 1]
        L, _ = ll(pte)
        w = " ".join(f"{c:+.2f}" for c in lr.coef_[0])
        print(f"  {name} | {lab:8}: log-loss {L:.4f}  (weights {w})")

# strategy filter head-to-head: 24h favorites, Elo vs GBM confirmation
print("\n=== 24h favorites strategy on joined 2025 sample: confirmation filter ===")
s = j.dropna(subset=["p0_24h"]).copy()
s = s[(s["volume"].fillna(0) >= 1000)]
for lab, pcol in [("Elo>=0.70", "p_elo_cal"), ("GBM>=0.70", "p_gbm"), ("no filter", None)]:
    pnls = []
    for r in s.itertuples():
        for q, won, pmod in [(r.p0_24h, r.winner_is_outcome0, None),
                             (1 - r.p0_24h, not r.winner_is_outcome0, None)]:
            if 0.80 <= q <= 0.97:
                pconf = (r.p_elo_cal if won else 1 - r.p_elo_cal) if pcol == "p_elo_cal" else \
                        (r.p_gbm if won else 1 - r.p_gbm) if pcol == "p_gbm" else 1.0
                if pconf >= 0.70:
                    qe = q + 0.01
                    pnls.append((1.0 - qe) / qe if won else -1.0)
                break
    if pnls:
        arr = np.array(pnls)
        print(f"  {lab:10}: n={len(arr):3d} hit={np.mean(arr > 0):.3f} "
              f"ROI={arr.mean()*100:+.1f}%  PnL={arr.sum():+.2f}")

out = pd.concat([M, X], axis=1)
out["p_gbm"] = np.nan
out.loc[ho, "p_gbm"] = p_full
out.loc[M["year"] >= 2025, "p_gbm"] = predict_sym(model, X[M["year"] >= 2025], FEATS)
out.to_parquet("data_raw/microstats_predictions.parquet")
print("\nsaved data_raw/microstats_predictions.parquet")
