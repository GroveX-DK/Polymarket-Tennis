import json, sys, io
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
c = Counter(); slugpfx = Counter()
n = 0
for l in open("data_raw/polymarket/events_tennis.jsonl", encoding="utf-8"):
    e = json.loads(l); n += 1
    slugpfx[e["slug"].split("-")[0]] += 1
    for m in e["markets"]:
        c[m.get("sportsMarketType")] += 1
print("events:", n)
print("market types:", dict(c.most_common()))
print("slug prefixes:", dict(slugpfx.most_common(15)))
