"""Validate the no-fit blend strategy: per-half stability + bet-level inspection."""
import json, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(11)

md = pd.read_parquet("data_raw/matched.parquet")
snaps = pd.DataFrame([json.loads(l) for l in open("data_raw/polymarket/price_snapshots.jsonl", encoding="utf-8")])
df = md.merge(snaps, on="market_id", how="inner")
df["game_start_ts"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
df = df[(df["td_comment"] == "Completed") & (df["winner_final_price"] > 0.5)]
df = df.dropna(subset=["p_elo_cal", "p0_24h", "p0_0h"]).copy()
df = df[(df["p0_24h"] > 0.03) & (df["p0_24h"] < 0.97)]
df["age24_h"] = ((df["game_start_ts"] - pd.Timestamp(0, tz="utc")).dt.total_seconds()
                 - 24 * 3600 - df["p0_24h_t"]) / 3600
df = df[df["age24_h"] <= 12]
df["y0"] = df["winner_is_outcome0"].astype(float)
df["p_elo0"] = np.where(df["winner_is_outcome0"], df["p_elo_cal"], 1 - df["p_elo_cal"])
df = df.sort_values("game_start_ts").reset_index(drop=True)

def logit(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def sigmoid(z): return 1 / (1 + np.exp(-z))
df["p_blend0"] = sigmoid(0.8 * logit(df["p0_24h"]) + 0.2 * logit(df["p_elo0"]))

def bets_for(t, thr, slip):
    out = []
    for r in t.itertuples():
        for (pb, q24, q0h, y, side) in [(r.p_blend0, r.p0_24h, r.p0_0h, r.y0, 0),
                                        (1 - r.p_blend0, 1 - r.p0_24h, 1 - r.p0_0h, 1 - r.y0, 1)]:
            qe = q24 + slip
            if pb - qe > thr and 0 < qe < 1:
                out.append({"ts": r.game_start_ts, "title": r.title, "side": side,
                            "outcome": r.outcome0 if side == 0 else r.outcome1,
                            "q24": q24, "q0h": q0h, "p_elo": r.p_elo0 if side == 0 else 1 - r.p_elo0,
                            "p_blend": pb, "won": y, "vol": r.volume,
                            "pnl_hold": (y - qe) / qe, "pnl_exit": ((q0h - slip) - qe) / qe})
                break
    return pd.DataFrame(out)

def summ(pnl):
    idxs = rng.integers(0, len(pnl), (4000, len(pnl)))
    rois = pnl.to_numpy()[idxs].mean(1)
    return pnl.mean(), *np.percentile(rois, [2.5, 97.5])

half = df["game_start_ts"].quantile(0.5)
print("=== per-half stability (no-fit blend, slip=0.01) ===")
for thr in [0.02, 0.03, 0.05]:
    for label, t in [("early", df[df["game_start_ts"] <= half]), ("late", df[df["game_start_ts"] > half])]:
        b = bets_for(t, thr, 0.01)
        if len(b) < 8:
            print(f" thr={thr} {label}: n={len(b)} (too few)")
            continue
        rh, lh, hh = summ(b["pnl_hold"]); rx, lx, hx = summ(b["pnl_exit"])
        print(f" thr={thr} {label:5}: n={len(b):3d} hit={b['won'].mean():.3f} | "
              f"hold {rh*100:6.1f}% [{lh*100:6.1f},{hh*100:6.1f}] | exit {rx*100:6.1f}% [{lx*100:6.1f},{hx*100:6.1f}]")

print("\n=== all thr=0.05 bets (slip=0.01) ===")
b = bets_for(df, 0.05, 0.01).sort_values("ts")
for r in b.itertuples():
    print(f"{str(r.ts)[:16]} {r.title[:48]:48} | bet {str(r.outcome)[:14]:14} q24={r.q24:.2f}->q0h={r.q0h:.2f} "
          f"elo={r.p_elo:.2f} {'WON ' if r.won else 'lost'} vol={r.vol:>9.0f} pnl_hold={r.pnl_hold:+.2f}")
print(f"\ntotal: n={len(b)} hold_pnl={b['pnl_hold'].sum():.2f} exit_pnl={b['pnl_exit'].sum():.2f} hit={b['won'].mean():.3f}")
print("bet-side distribution: dogs (q24<0.5):", (b["q24"] < 0.5).sum(), "| favs:", (b["q24"] >= 0.5).sum())
