import json, sys, io
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

md = pd.read_parquet("data_raw/matched.parquet")
md["year"] = pd.to_datetime(md["td_date"]).dt.year
print("matched by td year:")
print(md.groupby("year").agg(n=("market_id", "size"),
                             ps_missing=("PSW", lambda s: s.isna().mean().round(3)),
                             avg_missing=("AvgW", lambda s: s.isna().mean().round(3)),
                             b365_missing=("B365W", lambda s: s.isna().mean().round(3)),
                             completed=("td_comment", lambda s: (s == "Completed").mean().round(3))).to_string())

cur = pd.read_excel("data_raw/2026.xlsx")
print("\n2026.xlsx cols:", list(cur.columns))
print("rows:", len(cur))
