"""Serve/return micro-stats + context GBM model, v2.

v1 (items 4 & 5 of the accuracy list):
  #4 micro-stats (EWMA, halflife 25 matches, overall + per-surface):
     serve points won %, return points won %, ace rate, double-fault rate,
     hold %, break-point save %
  #5 context: best-of-5, round, indoor, home country, retirement/walkover risk
  base: overall + surface Elo (same K-schedule as build_elo), rank, rank points,
        age, height

v2 adds (items 1-3):
  #1 margin of victory: games-share Elo + EWMA game dominance, parsed from score
  #2 fatigue/schedule: court minutes last 7/14 days, matches last 14 days,
     days since last match (match dates approximated from tourney_date + round)
  #3 opponent-adjusted serve/return: SPW/RPW samples shifted by the opponent's
     return/serve strength vs the league running mean

All features computed strictly from matches BEFORE the one being predicted.
Model: HistGradientBoostingClassifier trained on 2000-2020, both orientations.
Eval: 2021-2024 holdout vs calibrated Elo and v1 on identical rows; 2025
Polymarket matched sample vs actual PM prices; strategy-filter head-to-head.

Data: TML-Database (Sackmann schema) via scripts/fetch_tml.py.
"""
import json
import os
import re
import sys
import unicodedata
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(7)

ROUND_ORD = {"RR": 0, "ER": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
             "QF": 5, "SF": 6, "BR": 7, "F": 8}
HL_STAT, HL_RET = 25, 60          # EWMA halflives (matches)
A_STAT = 1 - 0.5 ** (1 / HL_STAT)
A_RET = 1 - 0.5 ** (1 / HL_RET)
A_LG = 1 - 0.5 ** (1 / 5000)      # league-mean drift (era adjustment)
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

# ---------------- helpers ----------------
def parse_games(sc):
    """Total games won/lost from a score string; ignores super-TB brackets."""
    gw = gl = 0
    for tok in sc.split():
        tok = re.sub(r"\(\d*\)", "", tok)
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= 20 and b <= 20:
                gw += a
                gl += b
    return gw, gl


def match_day(date, rnd, draw):
    """Approximate calendar day of a match within its tournament."""
    dur = 13.0 if draw >= 96 else (8.0 if draw >= 48 else 6.0)
    return date + timedelta(days=round(rnd / 8.0 * dur))


def fnum(v):
    try:
        v = float(v)
        return v if v == v else np.nan
    except (ValueError, TypeError):
        return np.nan


# ---------------- chronological feature pass ----------------
class P:  # per-player state
    __slots__ = ("elo", "elo_s", "gelo", "n", "n_stat", "stats", "stats_s",
                 "ret", "recent")
    def __init__(self):
        self.elo, self.elo_s, self.gelo = 1500.0, {}, 1500.0
        self.n, self.n_stat = 0, 0
        self.stats = {}       # name -> ewma value (overall)
        self.stats_s = {}     # (name, surface) -> ewma value
        self.ret = 0.0
        self.recent = []      # [(match_day, minutes), ...] for fatigue

players = {}
def get(p):
    if p not in players:
        players[p] = P()
    return players[p]

def ew(d, k, x, a=A_STAT):
    d[k] = x if k not in d else (1 - a) * d[k] + a * x

STATS = ["spw", "rpw", "ace", "df", "hold", "bpsv"]
LG = {"spw": 0.645}  # league running mean of serve points won (era-adjusted)

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

def fatigue(p, mday):
    """(minutes last 7d, minutes last 14d, matches last 14d, days since last)."""
    m7 = m14 = n14 = 0.0
    last = None
    for d, mins in p.recent:
        gap = (mday - d).days
        if gap <= 0:
            continue
        if gap <= 14:
            m14 += mins
            n14 += 1
            if gap <= 7:
                m7 += mins
        if last is None or d > last:
            last = d
    dsl = (mday - last).days if last is not None else np.nan
    return m7, m14, n14, dsl

def feat_vec(pw, pl, surface, r, w_home, l_home, mday):
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
    rk_w, rk_l = fnum(r["winner_rank"]), fnum(r["loser_rank"])
    f["lrank_d"] = -(np.log(rk_w) - np.log(rk_l)) if rk_w > 0 and rk_l > 0 else np.nan
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
    # ---- v2: margin of victory ----
    f["gelo_d"] = pw.gelo - pl.gelo
    gs = pw.stats.get("gsh"), pl.stats.get("gsh")
    f["gsh_d"] = (gs[0] - gs[1]) if None not in gs else np.nan
    # ---- v2: opponent-adjusted serve/return ----
    for nm in ("aspw", "arpw"):
        v = pw.stats.get(nm), pl.stats.get(nm)
        f[nm + "_d"] = (v[0] - v[1]) if None not in v else np.nan
    # ---- v2: fatigue/schedule ----
    w7, w14, wn14, wdsl = fatigue(pw, mday)
    l7, l14, ln14, ldsl = fatigue(pl, mday)
    f["min7_d"] = w7 - l7
    f["min14_d"] = w14 - l14
    f["n14_d"] = wn14 - ln14
    f["dsl_d"] = (np.log1p(wdsl) - np.log1p(ldsl)) \
        if wdsl == wdsl and ldsl == ldsl else np.nan
    return f

FEATS_V1 = (["elo_d", "elo_s_d"] + [n + s for n in STATS for s in ("_d", "_s_d")]
            + ["ret_d", "lrank_d", "lpts_d", "age_d", "ht_d", "home_d",
               "bo5", "rnd", "indoor", "clay", "grass", "carpet"])

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
    draw = fnum(r["draw_size"])
    mday = match_day(r["date"], r["rnd"], draw if draw == draw else 32)

    if not walkover:
        host = HOME.get(r["tourney_name"])
        w_home = host is not None and r["winner_ioc"] == host
        l_home = host is not None and r["loser_ioc"] == host
        if (pw.n >= MIN_N and pl.n >= MIN_N
                and pw.n_stat >= MIN_STAT and pl.n_stat >= MIN_STAT):
            f = feat_vec(pw, pl, surface, r, w_home, l_home, mday)
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
    # margin of victory (skip retired matches - game share is distorted)
    if not retired:
        gw_, gl_ = parse_games(sc)
        if gw_ + gl_ >= 12:
            share = gw_ / (gw_ + gl_)
            gexp = 1.0 / (1.0 + 10.0 ** (-(pw.gelo - pl.gelo) / 400.0))
            pw.gelo += k_elo(pw.n) * (share - gexp)
            pl.gelo -= k_elo(pl.n) * (share - gexp)
            ew(pw.stats, "gsh", share)
            ew(pl.stats, "gsh", 1.0 - share)
    sw, sl = match_stats(r, "w"), match_stats(r, "l")
    if sw and sl:
        sw["rpw"], sl["rpw"] = 1.0 - sl["spw"], 1.0 - sw["spw"]
        # opponent-adjusted samples use PRE-match opponent strengths
        lg_spw = LG["spw"]
        lg_rpw = 1.0 - lg_spw
        opp_arpw_w = pl.stats.get("arpw", lg_rpw)   # returner faced by winner
        opp_aspw_w = pl.stats.get("aspw", lg_spw)   # server faced by winner
        opp_arpw_l = pw.stats.get("arpw", lg_rpw)
        opp_aspw_l = pw.stats.get("aspw", lg_spw)
        adj = {
            "w": {"aspw": sw["spw"] + (opp_arpw_w - lg_rpw),
                  "arpw": sw["rpw"] + (opp_aspw_w - lg_spw)},
            "l": {"aspw": sl["spw"] + (opp_arpw_l - lg_rpw),
                  "arpw": sl["rpw"] + (opp_aspw_l - lg_spw)},
        }
        LG["spw"] = (1 - A_LG) * LG["spw"] + A_LG * (sw["spw"] + sl["spw"]) / 2
        for p, s, side in ((pw, sw, "w"), (pl, sl, "l")):
            for kk, v in s.items():
                ew(p.stats, kk, v)
                ew(p.stats_s, (kk, surface), v)
            for kk, v in adj[side].items():
                ew(p.stats, kk, v)
            p.n_stat += 1
    pw.ret = (1 - A_RET) * pw.ret
    pl.ret = (1 - A_RET) * pl.ret + A_RET * (1.0 if retired else 0.0)
    mins = fnum(r["minutes"])
    mins = min(max(mins, 20.0), 360.0) if mins == mins else 95.0
    for p in (pw, pl):
        p.recent.append((mday, mins))
        if len(p.recent) > 25:
            p.recent = [(d, m) for d, m in p.recent if (mday - d).days <= 45]
    pw.n += 1
    pl.n += 1

X = pd.DataFrame(rows)
M = pd.DataFrame(meta)
FEATS_V2 = list(X.columns)
ANTI = [c for c in FEATS_V2 if c.endswith("_d")]
print(f"feature rows (matches): {len(X)}, v1 features: {len(FEATS_V1)}, "
      f"v2 features: {len(FEATS_V2)}")

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

model_v2 = fit(X[tr], FEATS_V2)
model_v1 = fit(X[tr], FEATS_V1)
p_v2 = predict_sym(model_v2, X[ho], FEATS_V2)
p_v1 = predict_sym(model_v1, X[ho], FEATS_V1)
p_elo = M.loc[ho, "p_elo_cal"].to_numpy()

def ll(p):  # log-loss of winner-side probs, with accuracy
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(np.log(p)), np.mean(p > 0.5)

print("\n=== holdout 2021-2024 (identical rows) ===")
for lab, p in [("calibrated Elo (report model)", p_elo),
               ("GBM v1 (micro-stats + context)", p_v1),
               ("GBM v2 (+margin/fatigue/opp-adj)", p_v2)]:
    L, acc = ll(p)
    print(f"{lab:34} log-loss {L:.4f}  accuracy {acc:.3f}")

idx = rng.integers(0, ho.sum(), (2000, ho.sum()))
for lab, pa, pb in [("v2 vs Elo", p_v2, p_elo), ("v2 vs v1", p_v2, p_v1)]:
    d = (-np.log(np.clip(pb, 1e-6, 1)) + np.log(np.clip(pa, 1e-6, 1)))
    diffs = d[idx].mean(1)
    print(f"improvement {lab}: {d.mean():+.4f} "
          f"[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}] "
          f"(positive = first model better)")

imp = permutation_importance(
    model_v2, *[pd.concat(orientations(X[ho].iloc[:2500][FEATS_V2]), ignore_index=True),
                np.r_[np.ones(2500), np.zeros(2500)]],
    n_repeats=3, random_state=0, scoring="neg_log_loss")
order = np.argsort(-imp.importances_mean)
print("\ntop feature importances (permutation, holdout):")
for i in order[:14]:
    print(f"  {FEATS_V2[i]:10} {imp.importances_mean[i]:.4f}")

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

is25 = M["year"] >= 2025
ev = M[is25].copy()
ev["p_gbm"] = predict_sym(model_v2, X[is25], FEATS_V2)
ev["p_gbm_v1"] = predict_sym(model_v1, X[is25], FEATS_V1)
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
    Lpm, _ = ll(pm_pw)
    L2, _ = ll(sub["p_gbm"].to_numpy())
    L1, _ = ll(sub["p_gbm_v1"].to_numpy())
    Le, _ = ll(sub["p_elo_cal"].to_numpy())
    print(f"{lab:12} n={len(sub):4d} | PM {Lpm:.4f} | GBMv2 {L2:.4f} | "
          f"GBMv1 {L1:.4f} | Elo {Le:.4f}")

# does the GBM v2 add information on top of the PM price? (chronological 2-fold)
sub = j.dropna(subset=["p0_0h"])
sub = sub[(sub["p0_0h"] > 0.01) & (sub["p0_0h"] < 0.99)].sort_values("td_date")
pm_pw = np.where(sub["winner_is_outcome0"], sub["p0_0h"], 1 - sub["p0_0h"])
lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
Z = np.c_[lo(pm_pw), lo(sub["p_gbm"].to_numpy())]
half = len(Z) // 2
print(f"\n=== does GBM v2 add info beyond the PM price at start? (n={len(Z)}) ===")
for name, tr_sl, te_sl in [("fit 1st half -> test 2nd", slice(0, half), slice(half, None)),
                           ("fit 2nd half -> test 1st", slice(half, None), slice(0, half))]:
    Ztr = np.r_[Z[tr_sl], -Z[tr_sl]]
    ytr = np.r_[np.ones(len(Z[tr_sl])), np.zeros(len(Z[tr_sl]))]
    for cols, lab in [([0], "PM only"), ([0, 1], "PM + GBM")]:
        lr = LogisticRegression(C=100.0, fit_intercept=False).fit(Ztr[:, cols], ytr)
        pte = lr.predict_proba(Z[te_sl][:, cols])[:, 1]
        L, _ = ll(pte)
        wts = " ".join(f"{c:+.2f}" for c in lr.coef_[0])
        print(f"  {name} | {lab:8}: log-loss {L:.4f}  (weights {wts})")

# strategy filter head-to-head: 24h favorites, Elo vs GBM v1 vs v2 confirmation
print("\n=== 24h favorites strategy on joined 2025 sample: confirmation filter ===")
s = j.dropna(subset=["p0_24h"]).copy()
s = s[(s["volume"].fillna(0) >= 1000)]
for lab, pcol in [("Elo>=0.70", "p_elo_cal"), ("GBMv1>=0.70", "p_gbm_v1"),
                  ("GBMv2>=0.70", "p_gbm"), ("no filter", None)]:
    pnls = []
    for r in s.itertuples():
        for q, won in [(r.p0_24h, r.winner_is_outcome0),
                       (1 - r.p0_24h, not r.winner_is_outcome0)]:
            if 0.80 <= q <= 0.97:
                pconf = 1.0 if pcol is None else \
                    (getattr(r, pcol) if won else 1 - getattr(r, pcol))
                if pconf >= 0.70:
                    qe = q + 0.01
                    pnls.append((1.0 - qe) / qe if won else -1.0)
                break
    if pnls:
        arr = np.array(pnls)
        print(f"  {lab:12}: n={len(arr):3d} hit={np.mean(arr > 0):.3f} "
              f"ROI={arr.mean()*100:+.1f}%  PnL={arr.sum():+.2f}")

out = pd.concat([M, X], axis=1)
out["p_gbm"] = np.nan
out["p_gbm_v1"] = np.nan
out.loc[ho, "p_gbm"] = p_v2
out.loc[ho, "p_gbm_v1"] = p_v1
out.loc[is25, "p_gbm"] = predict_sym(model_v2, X[is25], FEATS_V2)
out.loc[is25, "p_gbm_v1"] = predict_sym(model_v1, X[is25], FEATS_V1)
out.to_parquet("data_raw/microstats_predictions.parquet")
print("\nsaved data_raw/microstats_predictions.parquet")
