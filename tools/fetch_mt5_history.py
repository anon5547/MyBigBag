#!/usr/bin/env python3
"""
fetch_mt5_history.py -- pull bars out of a running MetaTrader 5 terminal,
in BROKER SERVER TIME, which is the only time base the EA will ever see.

Run this on the Windows machine that has the terminal open:

    python tools/fetch_mt5_history.py --symbol XAUUSD --bars 200000 --out XAUUSD_M5.csv

WHY NOT USE ANY OLD CSV
-----------------------
MetaTrader5.copy_rates_* returns timestamps in the SERVER's clock, and the
V5 feature contract encodes hour-of-day and day-of-week from exactly that
clock. If you train on a file exported in UTC while your broker runs
UTC+2/+3 with DST, then hour_sin/hour_cos are wrong on every row, by a
different amount in summer and winter. The model does not error; it just
learns nothing from four of its forty inputs and you never find out.

This script also refuses to hand you a file with duplicated or
out-of-order timestamps, which is what a broken export usually looks like.
"""
from __future__ import annotations

import argparse
import sys


TF = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="M5", choices=sorted(TF))
    ap.add_argument("--bars", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("pip install MetaTrader5  (Windows only, with the terminal installed)")
        return 2

    import pandas as pd

    if not mt5.initialize():
        print("mt5.initialize() failed:", mt5.last_error())
        return 1

    try:
        info = mt5.symbol_info(args.symbol)
        if info is None:
            print(f"symbol '{args.symbol}' not found. Available names are broker specific "
                  f"(XAUUSD, XAUUSDm, GOLD, XAUUSD.pro ...). Check Market Watch.")
            return 1
        if not info.visible:
            mt5.symbol_select(args.symbol, True)

        tf = getattr(mt5, TF[args.timeframe])
        rates = mt5.copy_rates_from_pos(args.symbol, tf, 0, args.bars)
        if rates is None or len(rates) == 0:
            print("copy_rates_from_pos returned nothing:", mt5.last_error())
            return 1

        df = pd.DataFrame(rates)
        # mt5 gives epoch seconds in SERVER time; do NOT localise or convert
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]

        dupes = int(df["time"].duplicated().sum())
        unsorted = int((df["time"].diff().dropna() <= pd.Timedelta(0)).sum())
        if dupes or unsorted:
            print(f"!! {dupes} duplicate and {unsorted} non-increasing timestamps; cleaning")
            df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

        median_spread = float(df["spread"].median())
        point = info.point
        print(f"terminal server time now: {pd.to_datetime(mt5.symbol_info_tick(args.symbol).time, unit='s')}")
        print(f"median spread: {median_spread:.0f} points  "
              f"= {median_spread * point:.3f} price units  (feed this to train_two_tier --cost)")

        df["time"] = df["time"].dt.strftime("%Y.%m.%d %H:%M:%S")
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} bars to {args.out}: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
