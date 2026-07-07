import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
lines = open("data_raw/polymarket/events_tennis.jsonl", encoding="utf-8").readlines()
print("events so far:", len(lines))
# find a singles match-looking event
shown = 0
for l in reversed(lines):
    e = json.loads(l)
    if " vs " in e["title"].replace("vs.", "vs") and "/" not in e["title"]:
        print(json.dumps(e, indent=1, ensure_ascii=False)[:1800])
        print("=" * 40)
        shown += 1
        if shown >= 2:
            break
