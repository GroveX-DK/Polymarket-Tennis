"""Survey closed tennis events on Polymarket Gamma API."""
import urllib.request, json, sys

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

evs = get("https://gamma-api.polymarket.com/events?limit=100&closed=true&tag_slug=tennis&order=id&ascending=false")
print(f"got {len(evs)} events")
for e in evs[:50]:
    print(e["id"], "|", e["slug"][:60], "|", e["title"][:45], "| vol:", round(e.get("volume", 0)))
