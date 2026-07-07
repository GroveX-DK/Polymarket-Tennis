"""Fetch pre-match price history from CLOB for matched markets.

For each matched market, fetch token0 price history (10-min fidelity) from
48h before scheduled start to 5 min after, and store snapshot prices at
lags: 24h, 6h, 1h, 0h (last point at/before each cutoff).
"""
import json, sys, io, time, urllib.request, urllib.parse, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

md = pd.read_parquet("data_raw/matched.parquet")
md["game_start_ts"] = pd.to_datetime(md["game_start"], utc=True, errors="coerce")
md = md.dropna(subset=["game_start_ts"]).reset_index(drop=True)
print("markets to fetch:", len(md))

OUT = "data_raw/polymarket/price_snapshots.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try:
            done.add(json.loads(l)["market_id"])
        except Exception:
            pass
print("already fetched:", len(done))

def fetch_one(row):
    t0 = int(row.game_start_ts.timestamp())
    url = ("https://clob.polymarket.com/prices-history?"
           + urllib.parse.urlencode({
               "market": row.token0, "startTs": t0 - 48 * 3600,
               "endTs": t0 + 300, "fidelity": 10}))
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                hist = json.load(r).get("history", [])
            break
        except Exception as e:
            if i == 3:
                return {"market_id": row.market_id, "error": str(e)}
            time.sleep(1.5 * (i + 1))
    snaps = {}
    for lag_name, lag_s in [("p0_24h", 24 * 3600), ("p0_6h", 6 * 3600),
                            ("p0_1h", 3600), ("p0_0h", 0)]:
        pts = [h for h in hist if h["t"] <= t0 - lag_s]
        snaps[lag_name] = pts[-1]["p"] if pts else None
        snaps[lag_name + "_t"] = pts[-1]["t"] if pts else None
    return {"market_id": row.market_id, "n_points": len(hist), **snaps}

todo = [r for r in md.itertuples() if r.market_id not in done]
print("fetching:", len(todo))
n_ok = 0
with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch_one, r): r.market_id for r in todo}
    for i, fut in enumerate(as_completed(futs)):
        res = fut.result()
        f.write(json.dumps(res) + "\n")
        if "error" not in res:
            n_ok += 1
        if (i + 1) % 250 == 0:
            f.flush()
            print(f"  {i+1}/{len(todo)} done ({n_ok} ok)")
print(f"DONE: {n_ok} ok / {len(todo)}")
