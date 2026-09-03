#!/usr/bin/env python3
"""
parity_check.py -- prove that the three feature implementations agree.

    A) MQL5/Include/SniperAI/*.mqh        (what the EA runs)
    B) tools/mql5_transcription.py        (a literal transcription of A)
    C) tools/sniper_features.py           (the fast pandas trainer path)

This script always checks B vs C, offline, with no MetaTrader needed.
If you also run MQL5/Scripts/SniperAI/ParityCheck.mq5 in the terminal it
writes MQL5/Files/SniperAI/parity_mql5.csv; pass --mql5 <that file> and
A vs C is checked too.

    python3 tools/parity_check.py
    python3 tools/parity_check.py --csv XAUUSD_M5.csv
    python3 tools/parity_check.py --mql5 parity_mql5.csv --csv XAUUSD_M5.csv

Exit code 0 = parity holds. Anything else = do not train, do not trade.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mql5_transcription import build_feature_vector          # noqa: E402
from sniper_features import (                                # noqa: E402
    FEATURE_NAMES,
    WARMUP_BARS,
    build_features,
)

REL_TOL = 1e-6
ABS_TOL = 1e-8


def synth_bars(n: int = 2200, seed: int = 20260903) -> pd.DataFrame:
    """A synthetic XAUUSD-ish M5 series: trends, gaps, spikes and flat spells."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-06-01 00:00:00")
    times = pd.date_range(t0, periods=n, freq="5min")

    drift = np.cumsum(rng.normal(0.0, 0.9, n))
    regime = 300.0 * np.sin(np.linspace(0, 6.0, n))
    close = 4200.0 + drift + regime

    # volatility clusters + a couple of shock bars
    vol = 0.6 + 1.4 * np.abs(np.sin(np.linspace(0, 11.0, n)))
    vol[1400:1404] *= 12.0
    high = close + np.abs(rng.normal(0, 1, n)) * vol
    low = close - np.abs(rng.normal(0, 1, n)) * vol
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.25, n)

    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])

    # a few dead bars: high == low. These break naive divisions.
    for k in (150, 151, 1600):
        high[k] = low[k] = close[k] = open_[k]

    volume = np.maximum(1, rng.poisson(180, n)).astype("float64")
    volume[1400:1404] *= 9.0
    volume[150] = 0                      # a zero-volume bar

    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
        }
    )


def check_contract_vs_mqh() -> bool:
    """
    The contract, the MQL5 enum and the MQL5 name table are three hand-written
    orderings of the same 40 features. One transposed line between them silently
    feeds every value into the wrong model column, and nothing raises.
    """
    import re

    repo = os.path.dirname(_HERE)
    path = os.path.join(repo, "MQL5", "Include", "SniperAI", "Features.mqh")
    if not os.path.exists(path):
        print(f"  ?? {path} not found; skipping")
        return True

    src = open(path, encoding="utf-8").read()
    ok = True

    try:
        block = src.split("static string names[SNP_FEATURE_DIM] =", 1)[1].split("};", 1)[0]
        mqh_names = re.findall(r'"([a-z0-9_]+)"', block)
    except IndexError:
        print("  !! cannot locate the name table in Features.mqh")
        return False

    if mqh_names != FEATURE_NAMES:
        ok = False
        print(f"  !! name table differs from the contract "
              f"({len(mqh_names)} vs {len(FEATURE_NAMES)})")
        for i, (a, b) in enumerate(zip(FEATURE_NAMES, mqh_names)):
            if a != b:
                print(f"     [{i}] contract={a!r}  Features.mqh={b!r}")

    try:
        enum_block = src.split("enum ENUM_SNP_FEATURE", 1)[1].split("};", 1)[0]
        enum_names = re.findall(r"\b(F_[A-Z0-9_]+)\b", enum_block)
    except IndexError:
        print("  !! cannot locate ENUM_SNP_FEATURE in Features.mqh")
        return False

    expect = ["F_" + n.upper() for n in FEATURE_NAMES]
    if enum_names != expect:
        ok = False
        print(f"  !! enum order differs from the contract "
              f"({len(enum_names)} vs {len(expect)})")
        for i, (a, b) in enumerate(zip(expect, enum_names)):
            if a != b:
                print(f"     [{i}] expected={a}  Features.mqh={b}")

    if ok:
        print(f"  contract == ENUM_SNP_FEATURE == name table  ({len(FEATURE_NAMES)} features)")
    return ok


def compare(name: str, a: np.ndarray, b: np.ndarray) -> tuple[bool, float, int]:
    denom = np.maximum(np.abs(a), np.abs(b))
    denom[denom < 1.0] = 1.0
    err = np.abs(a - b) / denom
    worst = float(np.nanmax(err)) if err.size else 0.0
    bad = int(np.sum(err > REL_TOL))
    return bad == 0, worst, bad


def check_transcription_vs_pandas(df: pd.DataFrame, probes: int = 40) -> bool:
    feats = build_features(df)
    n = len(df)
    idxs = np.linspace(WARMUP_BARS + 5, n - 1, probes).astype(int)
    idxs = sorted(set(int(i) for i in idxs if i < n))

    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    v = df["tick_volume"].to_numpy(dtype="float64")
    ts = pd.to_datetime(df["time"]).dt.to_pydatetime()

    ok_all = True
    worst_overall = 0.0
    worst_feature = ""

    for i in idxs:
        # the EA sees a rolling window ending at the signal bar
        s = max(0, i - WARMUP_BARS - 200 + 1)
        vec = build_feature_vector(o[s : i + 1], h[s : i + 1], lo[s : i + 1],
                                   c[s : i + 1], v[s : i + 1], list(ts[s : i + 1]))
        ref = feats.iloc[i].to_numpy(dtype="float64")
        got = np.asarray(vec, dtype="float64")

        for k, fname in enumerate(FEATURE_NAMES):
            a, b = ref[k], got[k]
            if not np.isfinite(a) or not np.isfinite(b):
                print(f"  !! non-finite {fname} at row {i}: pandas={a} transcription={b}")
                ok_all = False
                continue
            d = abs(a - b) / max(1.0, abs(a), abs(b))
            if d > worst_overall:
                worst_overall, worst_feature = d, fname
            if d > REL_TOL and abs(a - b) > ABS_TOL:
                print(f"  !! MISMATCH {fname} at row {i}: pandas={a!r} transcription={b!r} rel={d:.3e}")
                ok_all = False

    print(f"  probes={len(idxs)}  worst relative difference={worst_overall:.3e} ({worst_feature})")
    return ok_all


def check_mql5_dump(df: pd.DataFrame, dump_path: str) -> bool:
    dump = pd.read_csv(dump_path)
    if "bar_time" not in dump.columns:
        print("  !! dump has no bar_time column")
        return False
    feats = build_features(df)
    feats["bar_time"] = pd.to_datetime(df["time"]).dt.strftime("%Y.%m.%d %H:%M:%S")
    ref = feats.set_index("bar_time")

    ok_all = True
    matched = 0
    for _, row in dump.iterrows():
        key = str(row["bar_time"]).strip()
        if key not in ref.index:
            continue
        matched += 1
        for fname in FEATURE_NAMES:
            if fname not in dump.columns:
                print(f"  !! dump missing column {fname}")
                return False
            a = float(ref.loc[key, fname])
            b = float(row[fname])
            d = abs(a - b) / max(1.0, abs(a), abs(b))
            # MQL5 stores features as float32; loosen the tolerance accordingly
            if d > 2e-5:
                print(f"  !! MISMATCH {fname} @ {key}: python={a!r} mql5={b!r} rel={d:.3e}")
                ok_all = False
    print(f"  matched {matched} bars against the MetaTrader dump")
    return ok_all and matched > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="OHLCV csv (time,open,high,low,close,tick_volume)")
    ap.add_argument("--mql5", help="parity_mql5.csv written by ParityCheck.mq5")
    ap.add_argument("--probes", type=int, default=40)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        df.columns = [c.strip().lower() for c in df.columns]
        # MetaTrader writes "2026.09.01 13:35:00"
        try:
            df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M:%S")
        except (ValueError, TypeError):
            df["time"] = pd.to_datetime(df["time"])
        print(f"loaded {len(df)} bars from {args.csv}")
    else:
        df = synth_bars()
        print(f"using {len(df)} synthetic bars (no --csv given)")

    print("\n[1] feature ORDER: contract  vs  Features.mqh")
    ok0 = check_contract_vs_mqh()
    print("    ->", "PASS" if ok0 else "FAIL")

    print("\n[2] transcription of the MQL5 loops  vs  pandas trainer path")
    ok1 = check_transcription_vs_pandas(df, args.probes)
    print("    ->", "PASS" if ok1 else "FAIL")

    ok2 = True
    if args.mql5:
        print("\n[3] MetaTrader dump  vs  pandas trainer path")
        ok2 = check_mql5_dump(df, args.mql5)
        print("    ->", "PASS" if ok2 else "FAIL")
    else:
        print("\n[3] skipped (no --mql5 dump given). Run ParityCheck.mq5 in the")
        print("    terminal on the SAME bars to close the loop end to end.")

    print()
    if ok0 and ok1 and ok2:
        print("PARITY OK")
        return 0
    print("PARITY BROKEN -- do not train, do not trade")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
