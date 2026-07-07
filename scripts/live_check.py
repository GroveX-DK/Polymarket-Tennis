"""Live check: current PM prices for two requested matches + Elo coverage."""
import json, sys, io, urllib.request, urllib.parse
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

# --- find the live events ---
for q in ["Coulibaly Vasilev", "Palosi Piros"]:
    print("=" * 70)
    print("search:", q)
    res = get("https://gamma-api.polymarket.com/public-search?q=" + urllib.parse.quote(q) + "&limit_per_type=6")
    for e in res.get("events", []):
        if e.get("closed"):
            continue
        print(f"\nEVENT: {e['title']}  (slug={e['slug']})")
        for m in e.get("markets", []):
            smt = m.get("sportsMarketType", "")
            if smt and smt != "moneyline":
                continue
            outs = json.loads(m.get("outcomes") or "[]")
            print(f"  Q: {m.get('question')}")
            print(f"     outcomes: {outs}")
            print(f"     lastTrade: {m.get('lastTradePrice')}  bestBid: {m.get('bestBid')}  bestAsk: {m.get('bestAsk')}")
            print(f"     volume: {m.get('volumeNum')}  gameStart: {m.get('gameStartTime')}")

# --- Elo coverage for these players ---
print("\n" + "=" * 70)
print("Elo model coverage (tennis-data, tour-level only):")
elo = pd.read_parquet("data_raw/elo_predictions.parquet")
players = pd.concat([elo["Winner"], elo["Loser"]]).unique()
for name in ["Coulibaly", "Vasilev", "Palosi", "Piros"]:
    hits = [p for p in players if name.lower() in p.lower()]
    if not hits:
        print(f"  {name}: NOT in tennis-data at all -> model has no rating")
    else:
        for h in hits:
            n = ((elo["Winner"] == h) | (elo["Loser"] == h)).sum()
            last = elo.loc[(elo["Winner"] == h) | (elo["Loser"] == h), "Date"].max()
            print(f"  {name}: '{h}' -> {n} tour matches, last {str(last)[:10]}")
