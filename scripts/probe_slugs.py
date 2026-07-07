"""Probe slug conventions for grand-slam / masters ATP matches via public search."""
import urllib.request, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

for q in ["Alcaraz Sinner", "Sinner Djokovic", "Alcaraz Fritz"]:
    res = get("https://gamma-api.polymarket.com/public-search?q=" + urllib.parse.quote(q) + "&limit_per_type=20&events_status=closed")
    for e in res.get("events", []):
        print(e.get("slug"), "|", e.get("title"), "|", e.get("startDate", "")[:10])
    print("---")
