"""Download the TML-Database ATP season files (Sackmann-schema match stats).

Jeff Sackmann's original tennis_atp repo was taken down; Tennismylife/TML-Database
carries the same per-match serve/return columns (w_ace..l_bpFaced) and is updated
through the current season. ~80k matches, 2000-2026.

    python scripts/fetch_tml.py
"""
import os
import urllib.request

OUT = os.path.join("data_raw", "tml")
os.makedirs(OUT, exist_ok=True)
for year in range(2000, 2027):
    url = f"https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
    dst = os.path.join(OUT, f"{year}.csv")
    with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
        f.write(r.read())
    print(f"{year}: {os.path.getsize(dst)/1e3:.0f} KB")
