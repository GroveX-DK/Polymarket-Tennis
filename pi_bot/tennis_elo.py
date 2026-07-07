"""Pure-stdlib Elo engine + player-name matching for the Pi dry-run bot.

Replicates scripts/build_elo.py / scripts/live_scan.py exactly:
  - overall + per-surface Elo, K = 250 / (n_played + 5)^0.4
  - matches processed in (date, round-order) sequence
  - blended prob = sigmoid(0.6*d_overall + 0.4*d_surface) when surface known
  - calibration: shrink log-odds by scale 0.85 (fitted 2005-2020 in build_elo)

No numpy/pandas - runs comfortably in <50 MB RSS on a Raspberry Pi.
"""
import csv
import gzip
import io
import re
import unicodedata

ROUND_ORDER = {"1st Round": 0, "2nd Round": 1, "3rd Round": 2, "4th Round": 3,
               "Round Robin": 2, "Quarterfinals": 4, "Semifinals": 5, "The Final": 6}
CAL_SCALE = 0.85
MIN_MATCHES = 15  # both players need >=15 tour matches for a trusted prob


def _k(n):
    return 250.0 / ((n + 5) ** 0.4)


class EloBook:
    """Ratings state; feed matches chronologically via update()."""

    def __init__(self):
        self.elo = {}       # player -> overall rating
        self.elo_s = {}     # (player, surface) -> surface rating
        self.n = {}         # player -> matches played
        self.n_s = {}       # (player, surface) -> matches played on surface
        self.last_date = ""

    def update(self, winner, loser, surface, date=""):
        surface = surface or "Hard"
        ew, el = self.elo.get(winner, 1500.0), self.elo.get(loser, 1500.0)
        ews = self.elo_s.get((winner, surface), 1500.0)
        els = self.elo_s.get((loser, surface), 1500.0)
        exp = 1.0 / (1.0 + 10.0 ** (-(ew - el) / 400.0))
        exps = 1.0 / (1.0 + 10.0 ** (-(ews - els) / 400.0))
        nw, nl = self.n.get(winner, 0), self.n.get(loser, 0)
        self.elo[winner] = ew + _k(nw) * (1.0 - exp)
        self.elo[loser] = el - _k(nl) * (1.0 - exp)
        nws, nls = self.n_s.get((winner, surface), 0), self.n_s.get((loser, surface), 0)
        self.elo_s[(winner, surface)] = ews + _k(nws) * (1.0 - exps)
        self.elo_s[(loser, surface)] = els - _k(nls) * (1.0 - exps)
        self.n[winner], self.n[loser] = nw + 1, nl + 1
        self.n_s[(winner, surface)] = nws + 1
        self.n_s[(loser, surface)] = nls + 1
        if date > self.last_date:
            self.last_date = date

    def prob(self, p1, p2, surface=None):
        """Calibrated P(p1 beats p2), or None if either player is under-sampled."""
        if self.n.get(p1, 0) < MIN_MATCHES or self.n.get(p2, 0) < MIN_MATCHES:
            return None
        d = self.elo.get(p1, 1500.0) - self.elo.get(p2, 1500.0)
        if surface:
            ds = (self.elo_s.get((p1, surface), 1500.0)
                  - self.elo_s.get((p2, surface), 1500.0))
            d = 0.6 * d + 0.4 * ds
        # calibrated sigmoid: raw log10-odds is d/400, shrink by CAL_SCALE
        return 1.0 / (1.0 + 10.0 ** (-CAL_SCALE * d / 400.0))


def build_from_csv(path):
    """Build an EloBook from matches.csv.gz (date,round,winner,loser,surface)."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            rows.append((rec["date"], ROUND_ORDER.get(rec["round"], 1),
                         rec["winner"], rec["loser"], rec["surface"] or "Hard"))
    rows.sort(key=lambda r: (r[0], r[1]))
    book = EloBook()
    for date, _rnd, w, l, s in rows:
        book.update(w, l, s, date)
    return book


# ---------------- player-name matching (tennis-data "Surname I." style) -------------

def _norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", s.lower())


def _parse_td(name):
    """Split a tennis-data name into (normalized surname, first initial)."""
    toks = name.strip().split()
    inis = []
    while toks and re.fullmatch(r"([A-Z]\.)+[A-Z]?\.?", toks[-1]):
        inis.insert(0, toks.pop())
    return _norm("".join(toks)), _norm("".join(inis))[:1]


class NameIndex:
    """Match a Polymarket full name ('Carlos Alcaraz') to a tennis-data key
    ('Alcaraz C.') by surname suffix + first initial."""

    def __init__(self, players):
        self.sur_idx = {}
        for p in players:
            s, i = _parse_td(p)
            self.sur_idx.setdefault(s, []).append((p, i))

    def find(self, full_name):
        nf = _norm(full_name)
        parts = full_name.split()
        first_ini = _norm(parts[0])[:1] if parts else ""
        cands = []
        for s, lst in self.sur_idx.items():
            if s and nf.endswith(s):
                for p, i in lst:
                    if not i or i == first_ini:
                        cands.append((len(s), p))
        return max(cands)[1] if cands else None


# ---------------- tournament -> surface guess ----------------------------------------

_GRASS = ["wimbledon", "halle", "queen", "eastbourne", "stuttgart", "mallorca",
          "hertogenbosch", "newport", "boss open"]
_CLAY = ["roland garros", "french open", "monte carlo", "monte-carlo", "madrid",
         "rome", "italian open", "barcelona", "hamburg", "munich", "geneva", "lyon",
         "estoril", "bucharest", "marrakech", "houston", "kitzbuhel", "gstaad",
         "umag", "bastad", "buenos aires", "rio", "santiago", "cordoba", "bogota"]


def guess_surface(tournament):
    t = tournament.lower()
    for kw in _GRASS:
        if kw in t:
            return "Grass"
    for kw in _CLAY:
        if kw in t:
            return "Clay"
    return "Hard"


if __name__ == "__main__":
    import os
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "matches.csv.gz")
    book = build_from_csv(path)
    print(f"ratings through {book.last_date}: {len(book.elo)} players")
    top = sorted(book.elo.items(), key=lambda kv: -kv[1])[:10]
    for p, r in top:
        print(f"  {p:24} {r:7.1f}  (n={book.n[p]})")
