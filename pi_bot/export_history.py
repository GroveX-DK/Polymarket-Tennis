"""Export the minimal ATP match history the Pi bot needs to compute Elo ratings.

Run this on the development machine (needs pandas + the data_raw/ parquet files):

    python pi_bot/export_history.py

Writes two files under pi_bot/data/:
  matches.csv.gz     date,round,winner,loser,surface (~70k rows, ~0.5 MB)
  tournaments.json   ATP tour-level tournament/location keywords by month,
                     from the two most recent seasons - the bot uses this to
                     reject Challenger/ITF/WTA markets the backtest never
                     validated on.
The Pi only ever consumes these files (plus incremental updates fetched by
update_ratings.py) - it never needs pandas or parquet.
"""
import gzip
import json
import os
import re
import sys
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "matches.csv.gz")

hist = pd.read_parquet(os.path.join(ROOT, "data_raw", "tennisdata_all.parquet"))
cur = pd.read_excel(os.path.join(ROOT, "data_raw", "2026.xlsx"))
cols = ["Date", "Round", "Winner", "Loser", "Surface"]
df = pd.concat([hist[cols], cur[cols]], ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df = df.dropna(subset=["Winner", "Loser"])
df["Surface"] = df["Surface"].fillna("Hard")
df["Round"] = df["Round"].fillna("")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with gzip.open(OUT, "wt", encoding="utf-8", newline="") as f:
    f.write("date,round,winner,loser,surface\n")
    for r in df.itertuples(index=False):
        # player names can contain commas in theory; they don't in tennis-data,
        # but quote defensively anyway
        row = [str(r.Date), str(r.Round), str(r.Winner), str(r.Loser), str(r.Surface)]
        f.write(",".join('"%s"' % v.replace('"', '""') if "," in v else v for v in row) + "\n")

print(f"wrote {len(df)} matches ({df['Date'].min()} .. {df['Date'].max()}) -> {OUT}")
print(f"size: {os.path.getsize(OUT)/1e6:.2f} MB")


# ---- ATP tour calendar keywords (locations + tournament names, last 2 seasons) ----

def norm_kw(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

cal = pd.concat([hist[["Date", "Location", "Tournament"]],
                 cur[["Date", "Location", "Tournament"]]], ignore_index=True)
cal["Date"] = pd.to_datetime(cal["Date"])
cal = cal[cal["Date"].dt.year >= cal["Date"].dt.year.max() - 1]
months = {}
for r in cal.itertuples(index=False):
    for kw in {norm_kw(r.Location), norm_kw(r.Tournament)}:
        if len(kw) >= 4:
            months.setdefault(kw, set()).add(int(r.Date.month))
cal_out = os.path.join(os.path.dirname(OUT), "tournaments.json")
with open(cal_out, "w", encoding="utf-8") as f:
    json.dump({k: sorted(v) for k, v in sorted(months.items())}, f, indent=0)
print(f"wrote {len(months)} tour-level tournament keywords -> {cal_out}")
