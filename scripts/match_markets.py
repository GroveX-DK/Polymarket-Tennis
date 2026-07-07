"""Link Polymarket match-winner markets to tennis-data results & Elo predictions.

Output: data_raw/matched.parquet — one row per matched market with
token ids, outcome names, resolution, tennis-data odds and Elo probs.
"""
import json, sys, io, re, unicodedata
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", s.lower())

def parse_td_name(name):
    """tennis-data 'Bautista Agut R.' -> (surname_norm, first_initial)"""
    toks = name.strip().split()
    initials = []
    while toks and re.fullmatch(r"([A-Z]\.)+[A-Z]?\.?", toks[-1]):
        initials.insert(0, toks.pop())
    surname = norm("".join(toks))
    ini = norm("".join(initials))[:1]
    return surname, ini

# ---------- load elo predictions (match results) ----------
elo = pd.read_parquet("data_raw/elo_predictions.parquet")
elo = elo[elo["Date"] >= "2025-01-01"].copy()
elo["w_sur"], elo["w_ini"] = zip(*elo["Winner"].map(parse_td_name))
elo["l_sur"], elo["l_ini"] = zip(*elo["Loser"].map(parse_td_name))
elo["d"] = elo["Date"].dt.normalize()
print("tennis-data matches 2025+:", len(elo))

# index: (surname_pair frozenset) -> list of rows, allow date +/- 3d
from collections import defaultdict
idx = defaultdict(list)
for r in elo.itertuples():
    idx[frozenset((r.w_sur, r.l_sur))].append(r)

# ---------- load PM events, keep match-winner markets ----------
VS = re.compile(r"^\s*(.*?)\s*[:–-]\s*(.+?)\s+vs\.?\s+(.+?)\s*$", re.I)

rows = []
skipped_title = 0
for l in open("data_raw/polymarket/events_tennis.jsonl", encoding="utf-8"):
    e = json.loads(l)
    title = e["title"].replace("–", ":")
    m_title = VS.match(title)
    if not m_title:
        continue
    tourn, a, b = m_title.groups()
    if "/" in a or "/" in b:      # doubles
        continue
    for m in e["markets"]:
        smt = m.get("sportsMarketType")
        if smt not in ("moneyline", None):
            continue
        try:
            outcomes = json.loads(m["outcomes"] or "[]")
            prices = json.loads(m["outcomePrices"] or "[]")
            tokens = json.loads(m["clobTokenIds"] or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        if len(outcomes) != 2 or len(tokens) != 2 or len(prices) != 2:
            continue
        # match-winner markets have player-name outcomes (not Yes/No/Over)
        if smt is None:
            q = (m.get("question") or "")
            if outcomes[0] in ("Yes", "Over") or " vs" not in q.lower():
                continue
            # outcome names must correspond to the two players in the title
            o0, o1 = norm(outcomes[0]), norm(outcomes[1])
            na, nb = norm(a), norm(b)
            if not ((na.endswith(o0) or o0 in na) and (nb.endswith(o1) or o1 in nb)) and \
               not ((na.endswith(o1) or o1 in na) and (nb.endswith(o0) or o0 in nb)):
                continue
        rows.append({
            "event_id": e["id"], "event_slug": e["slug"], "title": e["title"],
            "tournament_pm": tourn, "player_a": a.strip(), "player_b": b.strip(),
            "market_id": m["id"], "condition_id": m["conditionId"],
            "outcome0": outcomes[0], "outcome1": outcomes[1],
            "price0_final": float(prices[0]), "price1_final": float(prices[1]),
            "token0": tokens[0], "token1": tokens[1],
            "game_start": m.get("gameStartTime"), "closed_time": m.get("closedTime"),
            "volume": m.get("volumeNum"), "smt": smt,
            "event_start": e.get("startDate"),
        })

pm = pd.DataFrame(rows)
# dedup: some events may have duplicate moneylines; keep highest volume
pm["volume"] = pm["volume"].fillna(0)
pm = pm.sort_values("volume", ascending=False).drop_duplicates("event_id").reset_index(drop=True)
print("PM match-winner markets:", len(pm))

# ---------- join to tennis-data ----------
def match_row(r):
    na, nb = norm(r.player_a), norm(r.player_b)
    ts = pd.to_datetime(r.game_start or r.event_start, utc=True, errors="coerce")
    if ts is pd.NaT:
        return None
    d = ts.tz_localize(None).normalize()
    best = None
    # try every td surname-pair contained in the PM full names
    for key, cand_rows in idx.items():
        k = tuple(key) if len(key) == 2 else (tuple(key)[0], tuple(key)[0])
        s1, s2 = k
        if not s1 or not s2:
            continue
        in_a = na.endswith(s1) or na.endswith(s2)
        in_b = nb.endswith(s1) or nb.endswith(s2)
        if not (in_a and in_b):
            continue
        if (na.endswith(s1) and nb.endswith(s1)) and s1 != s2:
            continue
        for c in cand_rows:
            if abs((c.d - d).days) <= 2:
                if best is None or abs((c.d - d).days) < abs((best.d - d).days):
                    best = c
    return best

matched = []
for r in pm.itertuples():
    c = match_row(r)
    if c is None:
        continue
    # which PM outcome is the tennis-data winner?
    o0, o1 = norm(r.outcome0), norm(r.outcome1)
    w_sur, l_sur = c.w_sur, c.l_sur
    if o0 == o1:
        continue
    def out_is(sur, o):
        return o == sur or o.endswith(sur) or sur.endswith(o)
    w0 = out_is(w_sur, o0) and not out_is(w_sur, o1)
    w1 = out_is(w_sur, o1) and not out_is(w_sur, o0)
    if not (w0 or w1):
        # fall back: match outcome to player_a/b then a/b to winner
        continue
    winner_token = r.token0 if w0 else r.token1
    winner_final = r.price0_final if w0 else r.price1_final
    matched.append({
        **r._asdict(),
        "td_date": c.Date, "td_winner": c.Winner, "td_loser": c.Loser,
        "td_tournament": c.Tournament, "td_series": c.Series, "td_surface": c.Surface,
        "td_comment": c.Comment,
        "p_elo_cal": c.p_elo_cal, "p_elo": c.p_elo,
        "n_w": c.n_w, "n_l": c.n_l,
        "PSW": c.PSW, "PSL": c.PSL, "AvgW": c.AvgW, "AvgL": c.AvgL,
        "B365W": c.B365W, "B365L": c.B365L, "MaxW": c.MaxW, "MaxL": c.MaxL,
        "winner_is_outcome0": w0,
        "winner_token": winner_token, "winner_final_price": winner_final,
    })

md = pd.DataFrame(matched).drop(columns=["Index"], errors="ignore")
print("matched to tennis-data:", len(md))
print("resolution sanity (winner final price ~1):", (md["winner_final_price"] > 0.5).mean())
print(md["td_series"].value_counts().to_string())
md.to_parquet("data_raw/matched.parquet")
print("saved data_raw/matched.parquet")
