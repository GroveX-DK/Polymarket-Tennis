"""Kelly staking simulation on the validated favorites strategy (147 bets).

Kelly fraction for buying a binary share at price q with believed prob p:
  f* = (p - q) / (1 - q)          (fraction of bankroll)
Two belief settings:
  A) p = Elo model prob (what "the model" believes)
  B) p = entry price + 4.5c (the historically measured favorite bias)
Bankroll starts at $1,000, bets processed chronologically.
"""
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
df = df[age <= 6].sort_values("game_start_ts")

SLIP = 0.01
bets = []
for r in df.itertuples():
    for (q, y, pe) in [(r.p0_24h, r.y0, r.p_elo0), (1 - r.p0_24h, 1 - r.y0, 1 - r.p_elo0)]:
        if 0.80 <= q <= 0.97 and pe >= 0.70:
            bets.append({"ts": r.game_start_ts, "q": q + SLIP, "won": y, "p_elo": pe})
            break
b = pd.DataFrame(bets).sort_values("ts").reset_index(drop=True)
print(f"bets: {len(b)}  hit: {b['won'].mean():.3f}  avg entry: {b['q'].mean():.3f}\n")

def simulate(belief_col_fn, kelly_mult, bank0=1000.0, cap=1.0):
    bank = bank0
    peak, maxdd = bank, 0.0
    skipped = 0
    for r in b.itertuples():
        p = belief_col_fn(r)
        q = r.q
        f = (p - q) / (1 - q)
        if f <= 0:
            skipped += 1
            continue
        f = min(f * kelly_mult, cap)
        stake = bank * f
        if r.won:
            bank += stake * (1 - q) / q
        else:
            bank -= stake
        peak = max(peak, bank)
        maxdd = max(maxdd, 1 - bank / peak)
        if bank < 1:
            return bank, maxdd, skipped, True
    return bank, maxdd, skipped, False

print("=== A) Kelly with the MODEL's belief (p = Elo prob) ===")
print(f"{'sizing':12} {'final bank':>12} {'return':>9} {'max DD':>8} {'skipped':>8} {'bust':>5}")
for mult, lab in [(1.0, "100% Kelly"), (0.5, "50% Kelly"), (0.25, "25% Kelly")]:
    bank, dd, sk, bust = simulate(lambda r: r.p_elo, mult)
    print(f"{lab:12} {bank:12,.0f} {bank/1000-1:9.1%} {dd:8.1%} {sk:8d} {'YES' if bust else 'no':>5}")

print("\n=== B) Kelly with the MEASURED bias (p = entry + 4.5c) ===")
for mult, lab in [(1.0, "100% Kelly"), (0.5, "50% Kelly"), (0.25, "25% Kelly")]:
    bank, dd, sk, bust = simulate(lambda r: min(r.q + 0.045, 0.995), mult)
    print(f"{lab:12} {bank:12,.0f} {bank/1000-1:9.1%} {dd:8.1%} {sk:8d} {'YES' if bust else 'no':>5}")

# flat benchmark
bank = 1000.0
peak, maxdd = bank, 0.0
for r in b.itertuples():
    stake = 50.0
    bank += stake * (1 - r.q) / r.q if r.won else -stake
    peak = max(peak, bank); maxdd = max(maxdd, 1 - bank / peak)
print(f"\nflat $50/bet benchmark: final {bank:,.0f} ({bank/1000-1:+.1%}), max DD {maxdd:.1%}")

# typical stake sizes at each setting
r0 = b.iloc[0]
print("\nexample stake on first bet (q={:.3f}, elo={:.2f}, $1k bank):".format(r0["q"], r0["p_elo"]))
for lab, p, mult in [("100% Kelly/Elo", r0["p_elo"], 1.0), ("25% Kelly/Elo", r0["p_elo"], 0.25),
                     ("100% Kelly/bias", min(r0["q"] + 0.045, 0.995), 1.0), ("25% Kelly/bias", min(r0["q"] + 0.045, 0.995), 0.25)]:
    f = max(0.0, (p - r0["q"]) / (1 - r0["q"])) * mult
    print(f"  {lab:16}: {f:6.1%} of bankroll = ${1000*f:,.0f}")
