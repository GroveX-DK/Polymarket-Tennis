"""Inspect the largest model-vs-PM divergences for data integrity."""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df = df[df["td_comment"] == "Completed"]
df = df[df["winner_final_price"] > 0.5]
df = df.dropna(subset=["AvgW", "AvgL", "p0_1h"])
df = df[(df["p0_1h"] > 0.02) & (df["p0_1h"] < 0.98)]

avg_w = (1 / df["AvgW"]) / (1 / df["AvgW"] + 1 / df["AvgL"])
df["p_avg0"] = np.where(df["winner_is_outcome0"], avg_w, 1 - avg_w)
df["y0"] = df["winner_is_outcome0"].astype(float)
df["edge0"] = df["p_avg0"] - df["p0_1h"]
df["abs_edge"] = df["edge0"].abs()
df["snap_age_h"] = (pd.to_datetime(df["game_start"], utc=True) -
                    pd.to_datetime(df["p0_1h_t"], unit="s", utc=True)).dt.total_seconds() / 3600

top = df.nlargest(30, "abs_edge")
for r in top.itertuples():
    side = r.outcome0 if r.edge0 > 0 else r.outcome1
    won = (r.y0 == 1) == (r.edge0 > 0)
    print(f"{str(r.td_date)[:10]} {r.title[:52]:52} | PM(o0)={r.p0_1h:.2f} Avg(o0)={r.p_avg0:.2f} "
          f"edge={r.edge0:+.2f} | AvgW={r.AvgW:.2f} AvgL={r.AvgL:.2f} | bet {side[:12]:12} "
          f"{'WON ' if won else 'lost'} | vol={r.volume:>9.0f} | snap_age={r.snap_age_h:.1f}h | {r.td_winner}")
