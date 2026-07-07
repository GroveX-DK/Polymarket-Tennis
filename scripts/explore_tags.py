"""Probe Gamma tags and event structure for ATP matches."""
import urllib.request, json

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

for slug in ["atp", "wta", "tennis", "grand-slam", "wimbledon"]:
    try:
        t = get(f"https://gamma-api.polymarket.com/tags/slug/{slug}")
        print(slug, "->", t.get("id"), t.get("label"))
    except Exception as e:
        print(slug, "ERR", e)

# sample an ATP event in detail
evs = get("https://gamma-api.polymarket.com/events?limit=3&closed=true&tag_slug=atp&order=id&ascending=false")
print(f"\natp tag events: {len(evs)}")
for e in evs:
    print("slug:", e["slug"], "| title:", e["title"])
    print("  startDate:", e.get("startDate"), "endDate:", e.get("endDate"))
    for m in e.get("markets", []):
        print("  market:", m.get("question"))
        print("    gameStartTime:", m.get("gameStartTime"), "| outcomes:", m.get("outcomes"), "| final prices:", m.get("outcomePrices"))
        print("    clobTokenIds:", str(m.get("clobTokenIds"))[:120])
        print("    volumeNum:", m.get("volumeNum"), "| closedTime:", m.get("closedTime"))
