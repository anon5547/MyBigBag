#!/usr/bin/env python3
"""
fetch_public_gold.py -- pull free public gold bars so the pipeline can be
smoke-tested WITHOUT a MetaTrader terminal.

    python3 tools/fetch_public_gold.py --interval 5m --range 60d  --out gold_5m.csv
    python3 tools/fetch_public_gold.py --interval 1h --range 730d --out gold_1h.csv

READ THIS BEFORE YOU TRAIN ANYTHING WITH IT
-------------------------------------------
This data is for exercising the CODE, not for producing a model you will
trade. Three reasons, none of them fixable here:

  1. GC=F is COMEX gold FUTURES. Your EA trades XAUUSD SPOT at a retail
     broker. They track each other but they are not the same instrument:
     futures have contract rolls, a different session, and no broker spread.
  2. Timestamps are UTC. The V5 time features encode BROKER SERVER time.
     Train on this and hour_sin/hour_cos are wrong by 2-3 hours forever.
  3. Volume here is exchange contract volume. Your EA sees tick volume.
     `vol_delta` and `rvol_20` mean different things in the two files.

For a model you intend to trade, use tools/fetch_mt5_history.py on the
Windows box running your terminal. There is no shortcut around that.

What this file IS good for: proving the feature engine, the labeller, the
CV splitter and the ONNX exporter all survive real market microstructure --
weekend gaps, session breaks, volatility clusters, thin overnight bars --
which synthetic random walks never reproduce.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

URL = ("https://query2.finance.yahoo.com/v8/finance/chart/"
       "{sym}?interval={interval}&range={rng}")

# Yahoo caps intraday history; asking for more silently returns less.
MAX_RANGE = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d",
             "30m": "60d", "1h": "730d", "1d": "max"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="GC=F", help="GC=F (COMEX gold futures)")
    ap.add_argument("--interval", default="5m", choices=sorted(MAX_RANGE))
    ap.add_argument("--range", dest="rng", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tz-shift-hours", type=float, default=0.0,
                    help="shift timestamps to imitate a broker server clock "
                         "(e.g. 3 for a UTC+3 server). Cosmetic -- it does not "
                         "make this data tradeable.")
    args = ap.parse_args()

    rng = args.rng or MAX_RANGE[args.interval]
    url = URL.format(sym=urllib.parse.quote(args.symbol), interval=args.interval, rng=rng)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as fh:
        payload = json.load(fh)

    result = payload.get("chart", {}).get("result")
    if not result:
        print("no data:", payload.get("chart", {}).get("error"))
        return 1

    import pandas as pd

    res = result[0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
        "tick_volume": q.get("volume"),
    })

    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["tick_volume"] = df["tick_volume"].fillna(0.0).astype("float64")
    # a real feed occasionally emits high < low or a zero print
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    if args.tz_shift_hours:
        df["time"] = df["time"] + pd.Timedelta(hours=args.tz_shift_hours)

    gaps = df["time"].diff().dropna()
    step = gaps.mode().iloc[0] if len(gaps) else pd.Timedelta(0)
    weekend_gaps = int((gaps > step * 12).sum())

    df["time"] = df["time"].dt.strftime("%Y.%m.%d %H:%M:%S")
    df.to_csv(args.out, index=False)

    print(f"{args.symbol} {args.interval} over {rng}")
    print(f"  raw rows {before} -> usable {len(df)}  (dropped {before - len(df)})")
    print(f"  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}"
          + (f"  (shifted {args.tz_shift_hours:+g}h)" if args.tz_shift_hours else "  (UTC)"))
    print(f"  modal bar step {step}, {weekend_gaps} session/weekend gaps")
    print(f"  wrote {args.out}")
    print("\n  REMINDER: pipeline smoke-test data only. Not for a model you will trade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
