"""Harvest all closed tennis events from Gamma API, windowed by start date
(offset pagination is capped at 2000, so we slice into daily windows)."""
import urllib.request, json, sys, time, io, os
from datetime import date, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {i+1} after error: {e} :: {url[:120]}")
            time.sleep(2 * (i + 1))
    raise RuntimeError("failed: " + url)

OUT = "data_raw/polymarket/events_tennis.jsonl"
os.makedirs("data_raw/polymarket", exist_ok=True)

seen = set()
total = 0
d0 = date(2024, 1, 1)
d1 = date(2026, 7, 3)

with open(OUT, "w", encoding="utf-8") as f:
    d = d0
    while d < d1:
        nxt = d + timedelta(days=1)
        offset = 0
        while True:
            url = (f"https://gamma-api.polymarket.com/events?limit=100&offset={offset}"
                   f"&closed=true&tag_slug=tennis&order=id&ascending=true"
                   f"&start_date_min={d.isoformat()}T00:00:00Z"
                   f"&start_date_max={nxt.isoformat()}T00:00:00Z")
            evs = get(url)
            if not evs:
                break
            for e in evs:
                if e["id"] in seen:
                    continue
                seen.add(e["id"])
                slim = {k: e.get(k) for k in
                        ["id", "slug", "title", "startDate", "endDate", "volume", "liquidity", "closed"]}
                slim["markets"] = [
                    {k: m.get(k) for k in
                     ["id", "question", "slug", "conditionId", "outcomes", "outcomePrices",
                      "clobTokenIds", "gameStartTime", "volumeNum", "closedTime", "startDate",
                      "endDate", "sportsMarketType"]}
                    for m in e.get("markets", [])]
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
                total += 1
            if len(evs) < 100:
                break
            offset += 100
            if offset >= 2000:
                print(f"  WARNING: window {d} hit offset cap!")
                break
            time.sleep(0.1)
        if d.day == 1:
            print(f"  {d}: cumulative {total} events")
        d = nxt

print(f"DONE: {total} events -> {OUT}")
