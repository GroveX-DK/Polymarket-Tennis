"""Weekly cadence of the final strategy: tournaments, bets and PnL per week."""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[(df["td_comment"] == "Completed") & (df["winner_final_price"] > 0.5)]
df = df.dropna(subset=["p0_24h", "p_elo_cal"]).copy()
df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])
df = df[df["volume"].fillna(0) >= 1000]
age = (df["game_start_ts"].astype("int64") / 1e9 - 24 * 3600 - df["p0_24h_t"]) / 3600
df = df[age <= 6]

SLIP = 0.01
rows = []
for r in df.itertuples():
    for (q, y, pe) in [(r.p0_24h, r.y0, r.p_elo0), (1 - r.p0_24h, 1 - r.y0, 1 - r.p_elo0)]:
        if 0.80 <= q <= 0.97 and pe >= 0.70:
            qe = q + SLIP
            rows.append({"ts": r.game_start_ts, "tourn": r.td_tournament,
                         "series": r.td_series, "pnl": (y - qe) / qe, "won": y})
            break
b = pd.DataFrame(rows).sort_values("ts")
b["week"] = b["ts"].dt.to_period("W")

wk = b.groupby("week").agg(bets=("pnl", "size"), tournaments=("tourn", "nunique"),
                           pnl=("pnl", "sum"), hit=("won", "mean"))
span_weeks = (b["ts"].max() - b["ts"].min()).days / 7
print(f"sample span: {b['ts'].min().date()} -> {b['ts'].max().date()}  (~{span_weeks:.0f} calendar weeks)")
print(f"weeks with at least one bet: {len(wk)}")
print(f"\nper ACTIVE week averages:")
print(f"  tournaments: {wk['tournaments'].mean():.2f} (median {wk['tournaments'].median():.0f}, max {wk['tournaments'].max()})")
print(f"  bets:        {wk['bets'].mean():.2f} (median {wk['bets'].median():.0f}, max {wk['bets'].max()})")
print(f"  PnL (units): {wk['pnl'].mean():+.3f} (median {wk['pnl'].median():+.3f})")
print(f"  losing weeks: {(wk['pnl'] < 0).sum()} of {len(wk)}")
print(f"\nper CALENDAR week (incl. empty): bets={len(b)/span_weeks:.2f}, PnL={b['pnl'].sum()/span_weeks:+.3f} units")

print("\nbets by tournament series:")
print(b.groupby("series").agg(bets=("pnl", "size"), pnl=("pnl", "sum"), hit=("won", "mean")).round(3).to_string())

print("\nweekly detail (last 20 active weeks):")
print(wk.tail(20).round(3).to_string())
