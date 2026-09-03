"""
sniper_features.py -- Python twin of MQL5/Include/SniperAI/{MathUtil,Features}.mqh

This file and the MQL5 headers are ONE artefact split across two languages.
If they drift, your model is trained on one distribution and traded on
another, and no amount of hyper-parameter tuning will save you.

Run tools/parity_check.py after ANY edit here.

Contract: tools/feature_contract.json
"""
from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
import pandas as pd

EPS = 1e-9

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "feature_contract.json"), "r", encoding="utf-8") as _fh:
    CONTRACT = json.load(_fh)

FEATURE_NAMES: list[str] = [f["name"] for f in CONTRACT["features"]]
FEATURE_DIM: int = CONTRACT["primary_dim"]
META_DIM: int = CONTRACT["meta_dim"]
META_EXTRA: list[str] = CONTRACT["meta_extra"]
WARMUP_BARS: int = CONTRACT["warmup_bars"]
META_NAMES: list[str] = FEATURE_NAMES + META_EXTRA

assert len(FEATURE_NAMES) == FEATURE_DIM, "contract is inconsistent"


# --------------------------------------------------------------------------
# primitives -- each one mirrors a function in MathUtil.mqh
# --------------------------------------------------------------------------
def sma(s: pd.Series, period: int) -> pd.Series:
    """SnpSMA: expanding mean until the window fills, then a true SMA."""
    return s.rolling(period, min_periods=1).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    """SnpEMA: ewm(span, adjust=False), seeded with the first value."""
    return s.ewm(span=period, adjust=False).mean()


def rma(s: pd.Series, period: int) -> pd.Series:
    """SnpRMA: Wilder smoothing == ewm(alpha=1/period, adjust=False)."""
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def rolling_std(s: pd.Series, period: int) -> pd.Series:
    """SnpRollingStd: POPULATION std (ddof=0), partial windows allowed."""
    return s.rolling(period, min_periods=1).std(ddof=0)


def rolling_max(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=1).max()


def rolling_min(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=1).min()


def safe_div(a, b, fallback=0.0):
    """SnpDiv: never propagates inf/nan."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full(np.broadcast(a, b).shape, float(fallback), dtype=np.float64)
    ok = np.abs(b) >= EPS
    np.divide(a, b, out=out, where=ok)
    out[~np.isfinite(out)] = float(fallback)
    return out


def rsi(close: pd.Series, period: int) -> pd.Series:
    """SnpRSI, Wilder."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta).clip(lower=0.0).fillna(0.0)
    ag = rma(gain, period)
    al = rma(loss, period)
    out = np.where(
        al.values < EPS,
        np.where(ag.values < EPS, 50.0, 100.0),
        100.0 - 100.0 / (1.0 + safe_div(ag.values, al.values, 0.0)),
    )
    return pd.Series(out, index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """SnpTrueRange. tr[0] = high[0] - low[0]."""
    prev = close.shift(1)
    a = high - low
    b = (high - prev).abs()
    c = (low - prev).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """SnpATR: Wilder RMA of TR -- NOT MetaTrader's iATR (which is an SMA)."""
    return rma(true_range(high, low, close), period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """SnpADX -> (adx, +di, -di), Wilder."""
    tr = true_range(high, low, close)
    up = high.diff()
    down = -low.diff()
    up.iloc[0] = 0.0
    down.iloc[0] = 0.0

    pdm = pd.Series(np.where((up > down) & (up > 0.0), up, 0.0), index=high.index)
    mdm = pd.Series(np.where((down > up) & (down > 0.0), down, 0.0), index=high.index)

    atr_w = rma(tr, period)
    spdm = rma(pdm, period)
    smdm = rma(mdm, period)

    pdi = 100.0 * safe_div(spdm.values, atr_w.values, 0.0)
    mdi = 100.0 * safe_div(smdm.values, atr_w.values, 0.0)
    dx = 100.0 * safe_div(np.abs(pdi - mdi), pdi + mdi, 0.0)

    pdi = pd.Series(pdi, index=high.index)
    mdi = pd.Series(mdi, index=high.index)
    adx_s = rma(pd.Series(dx, index=high.index), period)
    return adx_s, pdi, mdi


# --------------------------------------------------------------------------
# the feature frame
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """
    df must contain: time, open, high, low, close, tick_volume
    (`volume` and `real_volume` are accepted aliases for tick_volume).

    `time` MUST be the bar OPEN time in BROKER SERVER TIME. If you train on
    UTC and trade on a UTC+3 server, hour_sin/hour_cos are wrong by three
    hours on every single row and the model quietly learns nothing.

    Returns a DataFrame indexed like df with exactly FEATURE_NAMES columns.
    Rows before the warm-up are returned but must be dropped by the caller.
    """
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}
    if "tick_volume" not in cols:
        for alias in ("volume", "real_volume", "tickvol"):
            if alias in cols:
                d["tick_volume"] = d[cols[alias]]
                break
    for need in ("open", "high", "low", "close", "tick_volume"):
        if need not in d.columns:
            raise KeyError(f"missing column '{need}'; got {list(df.columns)}")

    ts = pd.to_datetime(d[time_col])
    o = d["open"].astype("float64")
    h = d["high"].astype("float64")
    lo = d["low"].astype("float64")
    c = d["close"].astype("float64")
    v = d["tick_volume"].astype("float64")

    out = pd.DataFrame(index=d.index)

    # ---- returns ---------------------------------------------------
    out["ret_1"] = np.log(safe_div(c.values, c.shift(1).values, 1.0))
    out["ret_3"] = np.log(safe_div(c.values, c.shift(3).values, 1.0))
    out["ret_5"] = np.log(safe_div(c.values, c.shift(5).values, 1.0))
    out["ret_10"] = np.log(safe_div(c.values, c.shift(10).values, 1.0))
    out["ret_1_lag1"] = out["ret_1"].shift(1)
    out["ret_1_lag2"] = out["ret_1"].shift(2)
    out["ret_1_lag3"] = out["ret_1"].shift(3)

    # ---- momentum ---------------------------------------------------
    out["rsi_7"] = rsi(c, 7)
    r14 = rsi(c, 14)
    out["rsi_14"] = r14
    out["rsi_14_lag1"] = r14.shift(1)

    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig = ema(macd_line, 9)
    macd_hist = macd_line - macd_sig
    # price-normalised: a raw MACD at gold 1800 and gold 4300 are different animals
    out["macd"] = 1000.0 * safe_div(macd_line.values, c.values, 0.0)
    out["macd_signal"] = 1000.0 * safe_div(macd_sig.values, c.values, 0.0)
    out["macd_hist"] = 1000.0 * safe_div(macd_hist.values, c.values, 0.0)
    out["macd_hist_lag1"] = 1000.0 * safe_div(
        macd_hist.shift(1).values, c.shift(1).values, 0.0
    )

    # ---- trend ------------------------------------------------------
    adx_s, pdi, mdi = adx(h, lo, c, 14)
    out["adx_14"] = adx_s
    out["plus_di_14"] = pdi
    out["minus_di_14"] = mdi

    ema20 = ema(c, 20)
    sma60 = sma(c, 60)
    out["ema20_dist"] = safe_div((c - ema20).values, c.values, 0.0)
    out["sma60_dist"] = safe_div((c - sma60).values, c.values, 0.0)
    out["ema20_slope"] = safe_div((ema20 - ema20.shift(5)).values, c.values, 0.0)

    # ---- volatility --------------------------------------------------
    atr14 = atr(h, lo, c, 14)
    atr50 = atr(h, lo, c, 50)
    out["atr14_norm"] = safe_div(atr14.values, c.values, 0.0)
    out["atr_ratio"] = safe_div(atr14.values, atr50.values, 1.0)

    sma20 = sma(c, 20)
    sd20 = rolling_std(c, 20)
    bb_up = sma20 + 2.0 * sd20
    bb_dn = sma20 - 2.0 * sd20
    out["bb_pctb"] = safe_div((c - bb_dn).values, (bb_up - bb_dn).values, 0.5)
    out["bb_width"] = safe_div((bb_up - bb_dn).values, sma20.values, 0.0)

    # ---- structure ---------------------------------------------------
    dc_up = rolling_max(h, 20)
    dc_dn = rolling_min(lo, 20)
    out["dist_to_res"] = safe_div((dc_up - c).values, c.values, 0.0)
    out["dist_to_sup"] = safe_div((c - dc_dn).values, c.values, 0.0)
    out["donchian_width"] = safe_div((dc_up - dc_dn).values, c.values, 0.0)
    out["close_zscore_20"] = safe_div((c - sma20).values, sd20.values, 0.0)

    # ---- candle ------------------------------------------------------
    rng = (h - lo)
    out["range_pct"] = safe_div(rng.values, c.values, 0.0)
    out["body_pct"] = safe_div((c - o).values, rng.values, 0.0)
    out["upper_wick_pct"] = safe_div(
        (h - pd.concat([o, c], axis=1).max(axis=1)).values, rng.values, 0.0
    )
    out["lower_wick_pct"] = safe_div(
        (pd.concat([o, c], axis=1).min(axis=1) - lo).values, rng.values, 0.0
    )

    # ---- order flow (TICK volume -- see docs/REVIEW.md) ---------------
    out["vol_delta"] = safe_div((v - v.shift(1)).values, (v.shift(1) + 1.0).values, 0.0)
    out["rvol_20"] = safe_div(v.values, sma(v, 20).values, 1.0)

    # ---- stochastic ---------------------------------------------------
    hh14 = rolling_max(h, 14)
    ll14 = rolling_min(lo, 14)
    stoch_k = 100.0 * safe_div((c - ll14).values, (hh14 - ll14).values, 0.5)
    stoch_k = pd.Series(stoch_k, index=d.index)
    out["stoch_k_14"] = stoch_k
    out["stoch_d_3"] = sma(stoch_k, 3)

    # ---- time (BROKER server time; MetaTrader weekday, 0 = Sunday) ----
    frac_hour = ts.dt.hour + ts.dt.minute / 60.0
    dow_mt5 = (ts.dt.dayofweek + 1) % 7          # pandas Mon=0 -> MT5 Sun=0
    out["hour_sin"] = np.sin(2.0 * np.pi * frac_hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * frac_hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow_mt5 / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow_mt5 / 7.0)

    out = out[FEATURE_NAMES]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_meta_features(
    base: pd.DataFrame, p_up: Sequence[float], side: Sequence[float]
) -> pd.DataFrame:
    """CSnpFeatureEngine::BuildMeta -- 40 base features + 3 primary outputs."""
    p_up = np.asarray(p_up, dtype=np.float64)
    side = np.asarray(side, dtype=np.float64)
    m = base.copy()
    m["primary_p_up"] = p_up
    m["side"] = side
    m["primary_edge"] = np.abs(2.0 * p_up - 1.0)
    return m[META_NAMES]


def drop_warmup(frame: pd.DataFrame, bars: int = WARMUP_BARS) -> pd.DataFrame:
    """The first `bars` rows are contaminated by indicator seeding. Bin them."""
    return frame.iloc[bars:].dropna()
