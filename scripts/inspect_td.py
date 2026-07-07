import sys, io
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

elo = pd.read_parquet("data_raw/elo_predictions.parquet")
cols = ["Date", "Tournament", "Round", "Winner", "Loser", "AvgW", "AvgL", "PSW", "PSL", "B365W", "B365L", "Comment"]

m1 = elo[(elo["Winner"].str.contains("Auger") & elo["Loser"].str.contains("Zverev")) |
         (elo["Winner"].str.contains("Zverev") & elo["Loser"].str.contains("Auger"))]
print(m1[m1["Date"] > "2025-11-01"][cols].to_string())
print()
m2 = elo[((elo["Winner"].str.contains("Bublik") & elo["Loser"].str.contains("Tsitsipas")) |
          (elo["Winner"].str.contains("Tsitsipas") & elo["Loser"].str.contains("Bublik")))]
print(m2[m2["Date"] > "2026-04-01"][cols].to_string())
print()
m3 = elo[((elo["Winner"].str.contains("Cilic") & elo["Loser"].str.contains("Borges")) |
          (elo["Winner"].str.contains("Borges") & elo["Loser"].str.contains("Cilic")))]
print(m3[m3["Date"] > "2026-06-01"][cols].to_string())
print()
m4 = elo[((elo["Winner"].str.contains("Shelton") & elo["Loser"].str.contains("Giron")) |
          (elo["Winner"].str.contains("Giron") & elo["Loser"].str.contains("Shelton")))]
print(m4[m4["Date"] > "2026-06-01"][cols].to_string())
