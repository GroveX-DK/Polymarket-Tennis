import json, sys, io, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

for q in ["Mensik Dimitrov", "Lehecka Molcan"]:
    res = get("https://gamma-api.polymarket.com/public-search?q=" + urllib.parse.quote(q) + "&limit_per_type=5")
    for e in res.get("events", []):
        if not e.get("closed"):
            print(f"{e['title']}\n  https://polymarket.com/event/{e['slug']}\n")
