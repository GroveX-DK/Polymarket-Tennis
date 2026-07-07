"""Pull full price curves for suspicious big-edge markets to check for
in-play contamination (match delayed vs scheduled gameStartTime)."""
import json, sys, io, urllib.request, urllib.parse
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
md = pd.read_parquet("data_raw/matched.parquet")

targets = [
    ("madrid-open-alexander-bublik", "Bublik vs Tsitsipas Madrid"),
    ("zverev", "Zverev vs FAA Finals"),
]
rows = []
for r in md.itertuples():
    t = r.title.lower()
    if ("bublik" in t and "tsitsipas" in t and "madrid" in t) or \
       ("zverev" in t and "auger" in t and "finals" in t.replace("world tour", "finals")):
        rows.append(r)

for r in rows:
    print("=" * 80)
    print(r.title, "| game_start:", r.game_start, "| closed:", r.closed_time, "| o0:", r.outcome0)
    t0 = int(pd.to_datetime(r.game_start, utc=True).timestamp())
    url = ("https://clob.polymarket.com/prices-history?" + urllib.parse.urlencode({
        "market": r.token0, "startTs": t0 - 24 * 3600, "endTs": t0 + 12 * 3600, "fidelity": 30}))
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        hist = json.load(resp)["history"]
    for h in hist:
        dt = pd.to_datetime(h["t"], unit="s", utc=True)
        rel = (h["t"] - t0) / 3600
        print(f"  {dt}  rel={rel:+6.1f}h  p0={h['p']:.3f}")
