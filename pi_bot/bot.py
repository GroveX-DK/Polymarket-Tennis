"""24/7 Polymarket ATP dry-run bot for Raspberry Pi (pure stdlib, <60 MB RSS).

Paper-trades the one strategy that survived the 2025-2026 backtest (REPORT.md):

    Buy the Polymarket favorite ~24h before match start when
      - its taker (ask) price is in [0.80, 0.97]
      - market volume >= $1,000 and the book is live (tight spread, recent trade)
      - the calibrated surface-blended Elo model also gives that side >= 0.70

Backtest result: +5.3% ROI, 147 bets, 95% CI [+0.3%, +9.7%] at 1c slippage.

Usage:
    python bot.py            # run forever, one scan every POLL_MINUTES
    python bot.py --once     # single scan cycle (cron-friendly / testing)
    python bot.py report     # print ledger summary and exit

State lives in pi_bot/data/paper_trades.sqlite; ratings come from
pi_bot/data/matches.csv.gz (refresh weekly with update_ratings.py).
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tennis_elo

# ---------------- strategy parameters (validated in REPORT.md) ----------------
PRICE_LO, PRICE_HI = 0.80, 0.97   # favorite ask-price bucket
ELO_MIN = 0.70                    # model confirmation threshold
VOL_MIN = 1000.0                  # market lifetime volume floor ($)
HRS_MIN, HRS_MAX = 18.0, 30.0     # entry window around the validated 24h snapshot
MAX_SPREAD = 0.05                 # skip stale/wide books (backtest used <=6h-old trades)
STAKE = 1.0                       # flat $ per paper bet
POLL_MINUTES = 30

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "paper_trades.sqlite")
MATCHES_CSV = os.path.join(DATA_DIR, "matches.csv.gz")
CALENDAR_JSON = os.path.join(DATA_DIR, "tournaments.json")
GAMMA = "https://gamma-api.polymarket.com"

VS = re.compile(r"^\s*(.*?)\s*[:–-]\s*(.+?)\s+vs\.?\s+(.+?)\s*$", re.I)
NOT_TOUR = ("wta", "itf", "junior", "challenger", "doubles", "exhibition",
            "utr", "uts", "ncaa", "college")


def log(msg):
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z  {msg}", flush=True)


def http_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pi-drybot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                log(f"HTTP failed ({e}): {url}")
            time.sleep(2 * (attempt + 1))
    return None


def parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------- ledger ----------------

def open_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS trades(
        market_id     TEXT PRIMARY KEY,
        entered_utc   TEXT NOT NULL,
        game_start_utc TEXT,
        tournament    TEXT,
        match_title   TEXT,
        side          TEXT,
        outcome_index INTEGER,
        price         REAL,
        bid           REAL,
        ask           REAL,
        model_p       REAL,
        volume        REAL,
        hours_out     REAL,
        status        TEXT DEFAULT 'open',
        pnl           REAL,
        resolved_utc  TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS skips(
        market_id TEXT PRIMARY KEY, reason TEXT, seen_utc TEXT)""")
    db.commit()
    return db


# ---------------- ratings (auto-reload when the csv is refreshed) ----------------

class Ratings:
    def __init__(self):
        self.mtime = None
        self.book = None
        self.names = None
        self.calendar = {}
        self.reload_if_stale()

    def reload_if_stale(self):
        m = os.path.getmtime(MATCHES_CSV)
        if m != self.mtime:
            t0 = time.time()
            self.book = tennis_elo.build_from_csv(MATCHES_CSV)
            self.names = tennis_elo.NameIndex(self.book.elo.keys())
            with open(CALENDAR_JSON, encoding="utf-8") as f:
                self.calendar = json.load(f)
            self.mtime = m
            log(f"ratings rebuilt through {self.book.last_date} "
                f"({len(self.book.elo)} players, "
                f"{len(self.calendar)} tour keywords, {time.time()-t0:.1f}s)")


def norm_kw(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def is_tour_level(tournament, calendar, month):
    """Only trade ATP tour-level events - the strategy was validated on nothing
    else. Accept if the title says ATP, or the tournament/location matches the
    recent tour calendar in this month +/- 1."""
    t = norm_kw(tournament)
    if any(x in t for x in NOT_TOUR):
        return False
    if "atp" in t.split() or "atp" in t:
        return True
    near = {month, month % 12 + 1, (month - 2) % 12 + 1}
    for kw, months in calendar.items():
        if kw in t and near.intersection(months):
            return True
    return False


# ---------------- scan: find qualifying entries ----------------

def fetch_open_tennis_events():
    events, offset = [], 0
    while True:
        batch = http_json(f"{GAMMA}/events?limit=100&offset={offset}"
                          f"&closed=false&tag_slug=tennis")
        if not batch:
            break
        events.extend(batch)
        if len(batch) < 100 or offset >= 1900:
            break
        offset += 100
    return events


def scan_for_entries(db, ratings):
    now = datetime.now(timezone.utc)
    events = fetch_open_tennis_events()
    if not events:
        log("no events returned (API hiccup?) - skipping entry scan")
        return 0
    placed = 0
    for ev in events:
        m_t = VS.match((ev.get("title") or "").replace("–", ":"))
        if not m_t:
            continue
        tourn, name_a, name_b = (g.strip() for g in m_t.groups())
        if "/" in name_a or "/" in name_b or "doubles" in (ev.get("slug") or ""):
            continue
        for mk in ev.get("markets", []):
            if mk.get("sportsMarketType") != "moneyline":
                continue
            mid = str(mk.get("id"))
            if db.execute("SELECT 1 FROM trades WHERE market_id=?", (mid,)).fetchone():
                continue
            gs = parse_iso(mk.get("gameStartTime"))
            if not gs:
                continue
            hrs_out = (gs - now).total_seconds() / 3600.0
            if not (HRS_MIN <= hrs_out <= HRS_MAX):
                continue
            bid, ask = mk.get("bestBid"), mk.get("bestAsk")
            vol = mk.get("volumeNum") or 0.0
            if bid is None or ask is None or vol < VOL_MIN:
                continue
            if (ask - bid) > MAX_SPREAD:
                continue  # wide/stale book: no fresh-trade guarantee
            if not is_tour_level(tourn, ratings.calendar, gs.month):
                continue  # Challenger/ITF/WTA etc. - never validated, skip
            try:
                outcomes = json.loads(mk["outcomes"])
            except Exception:
                continue
            if len(outcomes) != 2:
                continue
            # map outcome0/1 to the title's players; skip if ambiguous
            o0, o1 = tennis_elo._norm(outcomes[0]), tennis_elo._norm(outcomes[1])
            na, nb = tennis_elo._norm(name_a), tennis_elo._norm(name_b)
            hit = lambda o, n: bool(o) and (n.endswith(o) or o in n)
            a0, b0, a1, b1 = hit(o0, na), hit(o0, nb), hit(o1, na), hit(o1, nb)
            if a0 and b1 and not b0 and not a1:
                first, second = name_a, name_b
            elif b0 and a1 and not a0 and not b1:
                first, second = name_b, name_a
            else:
                continue  # can't map outcomes to players with certainty
            p1 = ratings.names.find(first)
            p2 = ratings.names.find(second)
            if not p1 or not p2:
                continue
            model_p0 = ratings.book.prob(p1, p2, tennis_elo.guess_surface(tourn))
            if model_p0 is None:
                continue
            mid_price = (bid + ask) / 2.0
            if mid_price >= 0.5:   # favorite = outcome0, buy at ask
                idx, side, buy, model_p = 0, first, ask, model_p0
            else:                  # favorite = outcome1, buy at 1-bid
                idx, side, buy, model_p = 1, second, round(1.0 - bid, 4), 1.0 - model_p0
            if not (PRICE_LO <= buy <= PRICE_HI):
                continue
            if model_p < ELO_MIN:
                db.execute("INSERT OR IGNORE INTO skips VALUES(?,?,?)",
                           (mid, f"elo {model_p:.2f} < {ELO_MIN}", now.isoformat()))
                continue
            db.execute(
                "INSERT INTO trades(market_id, entered_utc, game_start_utc, tournament,"
                " match_title, side, outcome_index, price, bid, ask, model_p, volume,"
                " hours_out) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, now.isoformat(), gs.isoformat(), tourn,
                 f"{name_a} vs {name_b}", side, idx, buy, bid, ask,
                 round(model_p, 4), vol, round(hrs_out, 2)))
            placed += 1
            log(f"PAPER BUY  {side} @ {buy:.3f}  ({tourn}: {name_a} vs {name_b}, "
                f"model {model_p:.2f}, vol ${vol:,.0f}, starts in {hrs_out:.0f}h)")
    db.commit()
    return placed


# ---------------- resolve finished trades ----------------

def resolve_open_trades(db):
    now = datetime.now(timezone.utc)
    rows = db.execute("SELECT market_id, game_start_utc, side, outcome_index, price "
                      "FROM trades WHERE status='open'").fetchall()
    resolved = 0
    for mid, gs_iso, side, idx, price in rows:
        gs = parse_iso(gs_iso)
        if gs and now < gs + timedelta(hours=2):
            continue  # match can't be over yet
        mk = http_json(f"{GAMMA}/markets/{mid}")
        if not mk or not mk.get("closed"):
            continue
        try:
            prices = [float(x) for x in json.loads(mk.get("outcomePrices") or "[]")]
        except Exception:
            prices = []
        if len(prices) != 2 or abs(prices[0] - prices[1]) < 0.9:
            status, pnl = "void", 0.0   # 50/50 resolution / refund / bad data
        elif prices[idx] > 0.5:
            status, pnl = "won", STAKE * (1.0 - price) / price
        else:
            status, pnl = "lost", -STAKE
        db.execute("UPDATE trades SET status=?, pnl=?, resolved_utc=? WHERE market_id=?",
                   (status, round(pnl, 4), now.isoformat(), mid))
        resolved += 1
        log(f"RESOLVED  {status.upper():4}  {side} @ {price:.3f}  pnl {pnl:+.3f}")
        time.sleep(0.3)  # be polite to the API
    db.commit()
    return resolved


# ---------------- reporting ----------------

def report(db):
    q = db.execute("SELECT status, COUNT(*), COALESCE(SUM(pnl),0) FROM trades "
                   "GROUP BY status").fetchall()
    counts = {s: (n, p) for s, n, p in q}
    n_open = counts.get("open", (0, 0))[0]
    n_won, pnl_won = counts.get("won", (0, 0.0))
    n_lost, pnl_lost = counts.get("lost", (0, 0.0))
    n_void = counts.get("void", (0, 0))[0]
    settled = n_won + n_lost
    total_pnl = pnl_won + pnl_lost
    print(f"paper ledger: {settled + n_open + n_void} trades "
          f"({n_open} open, {n_void} void)")
    if settled:
        print(f"settled: {settled}  hit {n_won/settled*100:.1f}%  "
              f"PnL {total_pnl:+.2f} units  ROI {total_pnl/(settled*STAKE)*100:+.1f}% "
              f"(flat ${STAKE:.0f} stakes, taker fills)")
    for row in db.execute(
            "SELECT entered_utc, match_title, side, price, model_p, status, pnl "
            "FROM trades ORDER BY entered_utc DESC LIMIT 15"):
        ent, title, side, price, mp, status, pnl = row
        pnl_s = f"{pnl:+.2f}" if pnl is not None else "  -  "
        print(f"  {ent[:16]}  {status:4}  {pnl_s}  {side} @ {price:.2f} "
              f"(model {mp:.2f})  [{title}]")


# ---------------- main loop ----------------

def cycle(db, ratings):
    ratings.reload_if_stale()
    placed = scan_for_entries(db, ratings)
    resolved = resolve_open_trades(db)
    n, pnl = db.execute("SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM trades "
                        "WHERE status IN ('won','lost')").fetchone()
    log(f"cycle done: +{placed} entries, {resolved} resolved; "
        f"lifetime {n} settled, PnL {pnl:+.2f} units")


def main():
    ap = argparse.ArgumentParser(description="Polymarket ATP dry-run paper trader")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("--once", action="store_true", help="single cycle then exit")
    args = ap.parse_args()

    db = open_db()
    if args.command == "report":
        report(db)
        return
    ratings = Ratings()
    log(f"strategy: fav ask [{PRICE_LO},{PRICE_HI}], elo>={ELO_MIN}, "
        f"vol>=${VOL_MIN:,.0f}, window {HRS_MIN}-{HRS_MAX}h, spread<={MAX_SPREAD}")
    while True:
        try:
            cycle(db, ratings)
        except Exception as e:
            log(f"cycle error: {e!r}")
        if args.once:
            break
        time.sleep(POLL_MINUTES * 60)


if __name__ == "__main__":
    main()
