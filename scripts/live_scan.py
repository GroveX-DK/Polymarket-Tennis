"""Scan live Polymarket tennis moneylines for model-vs-price mismatches.

1. Recompute Elo ratings through the latest tennis-data match.
2. Fetch active PM tennis events, keep upcoming moneylines (30min..72h out).
3. For matches where both players have >=15 tour matches, compute calibrated
   Elo prob and edge vs current ask; rank; flag validated-strategy qualifiers.
"""
import json, sys, io, re, time, unicodedata, urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------- 1. final Elo ratings ----------
hist = pd.read_parquet("data_raw/tennisdata_all.parquet")
cur = pd.read_excel("data_raw/2026.xlsx")
cur["season"] = 2026
cols = [c for c in hist.columns if c in cur.columns]
df = pd.concat([hist[cols], cur[cols]], ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"])
round_order = {"1st Round": 0, "2nd Round": 1, "3rd Round": 2, "4th Round": 3,
               "Round Robin": 2, "Quarterfinals": 4, "Semifinals": 5, "The Final": 6}
df["rnd"] = df["Round"].map(round_order).fillna(1)
df = df.sort_values(["Date", "rnd"], kind="stable").reset_index(drop=True)

elo, elo_s, n_played, n_s = {}, {}, {}, {}
def k(n): return 250.0 / ((n + 5) ** 0.4)
for w, l, s in zip(df["Winner"], df["Loser"], df["Surface"].fillna("Hard")):
    ew, el = elo.get(w, 1500.0), elo.get(l, 1500.0)
    ews, els = elo_s.get((w, s), 1500.0), elo_s.get((l, s), 1500.0)
    exp = 1 / (1 + 10 ** (-(ew - el) / 400))
    exps = 1 / (1 + 10 ** (-(ews - els) / 400))
    nw, nl = n_played.get(w, 0), n_played.get(l, 0)
    elo[w], elo[l] = ew + k(nw) * (1 - exp), el - k(nl) * (1 - exp)
    nws, nls = n_s.get((w, s), 0), n_s.get((l, s), 0)
    elo_s[(w, s)], elo_s[(l, s)] = ews + k(nws) * (1 - exps), els - k(nls) * (1 - exps)
    n_played[w], n_played[l] = nw + 1, nl + 1
    n_s[(w, s)], n_s[(l, s)] = nws + 1, nls + 1
print(f"ratings computed through {df['Date'].max().date()} for {len(elo)} players")

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", s.lower())

def parse_td(name):
    toks = name.strip().split()
    inis = []
    while toks and re.fullmatch(r"([A-Z]\.)+[A-Z]?\.?", toks[-1]):
        inis.insert(0, toks.pop())
    return norm("".join(toks)), norm("".join(inis))[:1]

sur_idx = {}
for p in elo:
    s, i = parse_td(p)
    sur_idx.setdefault(s, []).append((p, i))

def find_player(full):
    nf = norm(full)
    first_ini = norm(full.split()[0])[:1] if full.split() else ""
    cands = []
    for s, lst in sur_idx.items():
        if s and nf.endswith(s):
            for p, i in lst:
                if not i or i == first_ini:
                    cands.append((len(s), p))
    return max(cands)[1] if cands else None

# ---------- 2. live events ----------
def get(url, retries=3):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(1 + a)
    return []

evs, offset = [], 0
while True:
    batch = get(f"https://gamma-api.polymarket.com/events?limit=100&offset={offset}&closed=false&tag_slug=tennis")
    if not batch:
        break
    evs.extend(batch)
    if len(batch) < 100 or offset >= 1900:
        break
    offset += 100
print(f"active tennis events: {len(evs)}")

VS = re.compile(r"^\s*(.*?)\s*[:–-]\s*(.+?)\s+vs\.?\s+(.+?)\s*$", re.I)
now = datetime.now(timezone.utc)
rows = []
for e in evs:
    m_t = VS.match(e["title"].replace("–", ":"))
    if not m_t:
        continue
    tourn, a, b = m_t.groups()
    if "/" in a or "/" in b or "doubles" in e["slug"]:
        continue
    for m in e.get("markets", []):
        if m.get("sportsMarketType") != "moneyline":
            continue
        gs = pd.to_datetime(m.get("gameStartTime"), utc=True, errors="coerce")
        if gs is pd.NaT or not (now + timedelta(minutes=30) <= gs <= now + timedelta(hours=72)):
            continue
        try:
            outs = json.loads(m["outcomes"])
        except Exception:
            continue
        pa, pb = find_player(a), find_player(b)
        if not pa or not pb or n_played.get(pa, 0) < 15 or n_played.get(pb, 0) < 15:
            continue
        surface = "Grass" if any(x in tourn.lower() for x in ["wimbledon"]) else None
        d_all = elo[pa] - elo[pb]
        if surface:
            d_srf = elo_s.get((pa, surface), 1500) - elo_s.get((pb, surface), 1500)
            d = 0.6 * d_all + 0.4 * d_srf
        else:
            d = d_all
        p_raw = 1 / (1 + 10 ** (-d / 400))
        dex = np.log10(p_raw / (1 - p_raw))
        p_cal = 1 / (1 + 10 ** (-0.85 * dex))  # calibration scale from build_elo
        bid, ask = m.get("bestBid"), m.get("bestAsk")
        if bid is None or ask is None:
            continue
        # outcome0 assumed = player a (verify by name)
        o0 = norm(outs[0])
        if not norm(a).endswith(o0) and o0 not in norm(a):
            p_cal = 1 - p_cal  # outcomes flipped vs title
            pa, pb = pb, pa
        rows.append({
            "start": gs, "tourn": tourn.strip(), "match": f"{a.strip()} vs {b.strip()}",
            "p1_td": pa, "p2_td": pb, "n1": n_played.get(pa, 0), "n2": n_played.get(pb, 0),
            "elo_p1": p_cal, "bid1": bid, "ask1": ask,
            "vol": m.get("volumeNum") or 0, "hrs_out": (gs - now).total_seconds() / 3600,
        })

lv = pd.DataFrame(rows).drop_duplicates(subset=["match"])
print(f"upcoming tour-level matches with rated players: {len(lv)}\n")
if len(lv):
    # edge for buying side1 at ask1; side2 at (1-bid1)
    lv["edge_p1"] = lv["elo_p1"] - lv["ask1"]
    lv["edge_p2"] = (1 - lv["elo_p1"]) - (1 - lv["bid1"])
    lv["best_edge"] = lv[["edge_p1", "edge_p2"]].max(axis=1)
    lv["bet_side"] = np.where(lv["edge_p1"] >= lv["edge_p2"], 1, 2)
    lv = lv.sort_values("best_edge", ascending=False)
    print("=== biggest Elo-vs-price mismatches (buy at ask) ===")
    for r in lv.head(12).itertuples():
        side = r.match.split(" vs ")[r.bet_side - 1]
        price = r.ask1 if r.bet_side == 1 else round(1 - r.bid1, 3)
        pm_prob = (r.ask1 + r.bid1) / 2 if r.bet_side == 1 else 1 - (r.ask1 + r.bid1) / 2
        model_p = r.elo_p1 if r.bet_side == 1 else 1 - r.elo_p1
        print(f"{str(r.start)[:16]} | {r.tourn[:14]:14} | {r.match[:42]:42} | bet {side[:18]:18} "
              f"@ {price:.2f} | model {model_p:.2f} vs PM {pm_prob:.2f} | edge {r.best_edge:+.2f} | "
              f"vol ${r.vol:,.0f} | in {r.hrs_out:.0f}h | n=({r.n1},{r.n2})")

    print("\n=== VALIDATED-STRATEGY qualifiers (fav 0.80-0.97 ask, elo>=0.70, vol>=$1k, 6-48h out) ===")
    q = 0
    for r in lv.itertuples():
        for (price, model_p) in [(r.ask1, r.elo_p1), (round(1 - r.bid1, 3), 1 - r.elo_p1)]:
            if 0.80 <= price <= 0.97 and model_p >= 0.70 and r.vol >= 1000 and 6 <= r.hrs_out <= 48:
                side = r.match.split(" vs ")[0 if price == r.ask1 else 1]
                print(f"  BUY {side[:22]:22} @ {price:.3f} | {r.tourn[:14]} | {r.match[:40]} | "
                      f"model {model_p:.2f} | vol ${r.vol:,.0f} | starts in {r.hrs_out:.0f}h")
                q += 1
    if not q:
        print("  none right now")
