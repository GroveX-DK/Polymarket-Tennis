"""Refresh pi_bot/data/matches.csv.gz with the latest tennis-data.co.uk results.

Pure stdlib (parses the xlsx zip/XML by hand) so it runs on the Pi. Downloads the
current season's ATP file, replaces that season's rows in matches.csv.gz, and the
running bot picks up the new file automatically (mtime-based reload).

Only *results* (winner/loser/surface) are used - tennis-data's 2025/26 odds
columns are known-contaminated (REPORT.md) and are never read.

Run weekly, e.g. cron:  17 6 * * 1  python /home/pi/pi_bot/update_ratings.py
"""
import csv
import gzip
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "matches.csv.gz")
CALENDAR_JSON = os.path.join(DATA_DIR, "tournaments.json")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
WANTED = ["Date", "Round", "Winner", "Loser", "Surface", "Location", "Tournament"]


def season_url(year):
    return f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"


def col_letters(ref):
    return re.match(r"[A-Z]+", ref).group(0)


def col_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def excel_date(serial):
    return (date(1899, 12, 30) + timedelta(days=int(float(serial)))).isoformat()


def parse_xlsx(blob):
    """Yield dicts with WANTED columns from the first worksheet."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    shared = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    sheet_name = next(n for n in zf.namelist()
                      if re.fullmatch(r"xl/worksheets/sheet1\.xml", n))
    root = ET.fromstring(zf.read(sheet_name))
    header = {}
    for row in root.iter(f"{NS}row"):
        cells = {}
        for c in row.findall(f"{NS}c"):
            ref = c.get("r", "")
            v = c.find(f"{NS}v")
            if v is None or v.text is None:
                continue
            val = shared[int(v.text)] if c.get("t") == "s" else v.text
            cells[col_index(col_letters(ref))] = val
        if not header:
            header = {i: str(v).strip() for i, v in cells.items()}
            missing = [w for w in WANTED if w not in header.values()]
            if missing:
                raise RuntimeError(f"header row missing columns: {missing}")
            idx = {v: i for i, v in header.items()}
            continue
        rec = {w: cells.get(idx[w], "") for w in WANTED}
        if not rec["Winner"] or not rec["Loser"]:
            continue
        try:
            rec["Date"] = excel_date(rec["Date"])
        except (ValueError, TypeError):
            continue  # not a data row
        rec["Surface"] = rec["Surface"] or "Hard"
        yield rec


def main():
    year = datetime.now().year
    url = season_url(year)
    print(f"downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "pi-drybot/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    fresh = list(parse_xlsx(blob))
    if not fresh:
        print("no rows parsed - aborting, csv left untouched")
        sys.exit(1)
    print(f"parsed {len(fresh)} {year} matches "
          f"(latest: {max(r['Date'] for r in fresh)})")

    kept = []
    with gzip.open(CSV_PATH, "rt", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if not rec["date"].startswith(str(year)):
                kept.append(rec)
    n_old = len(kept)
    for r in fresh:
        kept.append({"date": r["Date"], "round": r["Round"], "winner": r["Winner"],
                     "loser": r["Loser"], "surface": r["Surface"]})

    tmp = CSV_PATH + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "round", "winner", "loser", "surface"])
        w.writeheader()
        w.writerows(kept)
    os.replace(tmp, CSV_PATH)
    print(f"matches.csv.gz updated: {n_old} historical + {len(fresh)} {year} rows")

    # merge this season's tournament/location keywords into the tour calendar
    def norm_kw(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    cal = {}
    if os.path.exists(CALENDAR_JSON):
        with open(CALENDAR_JSON, encoding="utf-8") as f:
            cal = {k: set(v) for k, v in json.load(f).items()}
    added = 0
    for r in fresh:
        month = int(r["Date"][5:7])
        for kw in {norm_kw(r.get("Location", "")), norm_kw(r.get("Tournament", ""))}:
            if len(kw) >= 4:
                if month not in cal.setdefault(kw, set()):
                    added += 1
                cal[kw].add(month)
    with open(CALENDAR_JSON, "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in sorted(cal.items())}, f, indent=0)
    print(f"tournaments.json: {len(cal)} keywords ({added} new month entries)")


if __name__ == "__main__":
    main()
