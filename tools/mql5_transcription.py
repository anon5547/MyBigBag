"""
mql5_transcription.py -- a LITERAL, loop-for-loop transcription of
MQL5/Include/SniperAI/{MathUtil,Features}.mqh into pure Python.

It is deliberately slow and un-pythonic. Its only job is to be obviously
the same code as the MQL5 headers, so that parity_check.py can prove the
fast pandas implementation in sniper_features.py agrees with what the EA
will actually compute at runtime.

Three implementations, two comparisons:
    MQL5 (.mqh)  ~=  this file  ~=  sniper_features.py
This file is the bridge you can execute without MetaTrader.
"""
from __future__ import annotations

import math
from typing import List, Sequence

EPS = 1e-9


def snp_div(a: float, b: float, fallback: float = 0.0) -> float:
    if abs(b) < EPS:
        return fallback
    r = a / b
    if not math.isfinite(r):
        return fallback
    return r


def snp_sma(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    dst = [0.0] * n
    run = 0.0
    for i in range(n):
        run += src[i]
        if i >= period:
            run -= src[i - period]
        cnt = i + 1 if i + 1 < period else period
        dst[i] = run / cnt
    return dst


def snp_ema(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    dst = [0.0] * n
    a = 2.0 / (period + 1.0)
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = a * src[i] + (1.0 - a) * dst[i - 1]
    return dst


def snp_rma(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    dst = [0.0] * n
    a = 1.0 / period
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = a * src[i] + (1.0 - a) * dst[i - 1]
    return dst


def snp_rolling_std(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    dst = [0.0] * n
    s = 0.0
    s2 = 0.0
    for i in range(n):
        s += src[i]
        s2 += src[i] * src[i]
        if i >= period:
            s -= src[i - period]
            s2 -= src[i - period] * src[i - period]
        cnt = i + 1 if i + 1 < period else period
        mean = s / cnt
        var = (s2 / cnt) - mean * mean
        if var < 0.0:
            var = 0.0
        dst[i] = math.sqrt(var)
    return dst


def snp_rolling_max(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    dst = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        dst[i] = max(src[start : i + 1])
    return dst


def snp_rolling_min(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    dst = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        dst[i] = min(src[start : i + 1])
    return dst


def snp_rsi(close: Sequence[float], period: int) -> List[float]:
    n = len(close)
    gain = [0.0] * n
    loss = [0.0] * n
    for i in range(1, n):
        d = close[i] - close[i - 1]
        gain[i] = d if d > 0.0 else 0.0
        loss[i] = -d if d < 0.0 else 0.0
    ag = snp_rma(gain, period)
    al = snp_rma(loss, period)
    dst = [0.0] * n
    for i in range(n):
        if al[i] < EPS:
            dst[i] = 50.0 if ag[i] < EPS else 100.0
        else:
            dst[i] = 100.0 - (100.0 / (1.0 + ag[i] / al[i]))
    return dst


def snp_true_range(high, low, close) -> List[float]:
    n = len(close)
    dst = [0.0] * n
    dst[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        dst[i] = max(a, b, c)
    return dst


def snp_atr(high, low, close, period) -> List[float]:
    return snp_rma(snp_true_range(high, low, close), period)


def snp_adx(high, low, close, period):
    n = len(close)
    tr = snp_true_range(high, low, close)
    pdm = [0.0] * n
    mdm = [0.0] * n
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up if (up > down and up > 0.0) else 0.0
        mdm[i] = down if (down > up and down > 0.0) else 0.0
    atr_w = snp_rma(tr, period)
    spdm = snp_rma(pdm, period)
    smdm = snp_rma(mdm, period)
    pdi = [0.0] * n
    mdi = [0.0] * n
    dx = [0.0] * n
    for i in range(n):
        pdi[i] = 100.0 * snp_div(spdm[i], atr_w[i], 0.0)
        mdi[i] = 100.0 * snp_div(smdm[i], atr_w[i], 0.0)
        dx[i] = 100.0 * snp_div(abs(pdi[i] - mdi[i]), pdi[i] + mdi[i], 0.0)
    return snp_rma(dx, period), pdi, mdi


def build_feature_vector(o, h, l, c, v, times) -> List[float]:
    """CSnpFeatureEngine::BuildFromArrays, transcribed. Returns 40 floats."""
    n = len(c)
    i = n - 1

    f_ret1 = math.log(snp_div(c[i], c[i - 1], 1.0))
    f_ret3 = math.log(snp_div(c[i], c[i - 3], 1.0))
    f_ret5 = math.log(snp_div(c[i], c[i - 5], 1.0))
    f_ret10 = math.log(snp_div(c[i], c[i - 10], 1.0))
    f_ret1_l1 = math.log(snp_div(c[i - 1], c[i - 2], 1.0))
    f_ret1_l2 = math.log(snp_div(c[i - 2], c[i - 3], 1.0))
    f_ret1_l3 = math.log(snp_div(c[i - 3], c[i - 4], 1.0))

    rsi7 = snp_rsi(c, 7)
    rsi14 = snp_rsi(c, 14)

    ema12 = snp_ema(c, 12)
    ema26 = snp_ema(c, 26)
    macd = [ema12[k] - ema26[k] for k in range(n)]
    macdsig = snp_ema(macd, 9)

    macd_n = 1000.0 * snp_div(macd[i], c[i])
    macdsig_n = 1000.0 * snp_div(macdsig[i], c[i])
    macdhist_n = 1000.0 * snp_div(macd[i] - macdsig[i], c[i])
    macdhist_l1 = 1000.0 * snp_div(macd[i - 1] - macdsig[i - 1], c[i - 1])

    adx, pdi, mdi = snp_adx(h, l, c, 14)

    ema20 = snp_ema(c, 20)
    sma60 = snp_sma(c, 60)
    ema20_dist = snp_div(c[i] - ema20[i], c[i])
    sma60_dist = snp_div(c[i] - sma60[i], c[i])
    ema20_slope = snp_div(ema20[i] - ema20[i - 5], c[i])

    atr14 = snp_atr(h, l, c, 14)
    atr50 = snp_atr(h, l, c, 50)
    atr14_norm = snp_div(atr14[i], c[i])
    atr_ratio = snp_div(atr14[i], atr50[i], 1.0)

    sd20 = snp_rolling_std(c, 20)
    sma20 = snp_sma(c, 20)
    bb_up = sma20[i] + 2.0 * sd20[i]
    bb_dn = sma20[i] - 2.0 * sd20[i]
    bb_pctb = snp_div(c[i] - bb_dn, bb_up - bb_dn, 0.5)
    bb_width = snp_div(bb_up - bb_dn, sma20[i])

    dcup = snp_rolling_max(h, 20)
    dcdn = snp_rolling_min(l, 20)
    dist_to_res = snp_div(dcup[i] - c[i], c[i])
    dist_to_sup = snp_div(c[i] - dcdn[i], c[i])
    donch_width = snp_div(dcup[i] - dcdn[i], c[i])
    close_z = snp_div(c[i] - sma20[i], sd20[i])

    rng = h[i] - l[i]
    range_pct = snp_div(rng, c[i])
    body_pct = snp_div(c[i] - o[i], rng)
    upper_wick_pct = snp_div(h[i] - max(o[i], c[i]), rng)
    lower_wick_pct = snp_div(min(o[i], c[i]) - l[i], rng)

    volsma20 = snp_sma(v, 20)
    vol_delta = snp_div(v[i] - v[i - 1], v[i - 1] + 1.0)
    rvol_20 = snp_div(v[i], volsma20[i], 1.0)

    hh14 = snp_rolling_max(h, 14)
    ll14 = snp_rolling_min(l, 14)
    stk = [100.0 * snp_div(c[k] - ll14[k], hh14[k] - ll14[k], 0.5) for k in range(n)]
    std3 = snp_sma(stk, 3)

    ts = times[i]
    frac_hour = ts.hour + ts.minute / 60.0
    dow = (ts.weekday() + 1) % 7          # python Mon=0 -> MetaTrader Sun=0
    hour_sin = math.sin(2.0 * math.pi * frac_hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * frac_hour / 24.0)
    dow_sin = math.sin(2.0 * math.pi * dow / 7.0)
    dow_cos = math.cos(2.0 * math.pi * dow / 7.0)

    return [
        f_ret1, f_ret3, f_ret5, f_ret10,
        f_ret1_l1, f_ret1_l2, f_ret1_l3,
        rsi7[i], rsi14[i], rsi14[i - 1],
        macd_n, macdsig_n, macdhist_n, macdhist_l1,
        adx[i], pdi[i], mdi[i],
        ema20_dist, sma60_dist, ema20_slope,
        atr14_norm, atr_ratio,
        bb_pctb, bb_width,
        dist_to_res, dist_to_sup, donch_width, close_z,
        range_pct, body_pct, upper_wick_pct, lower_wick_pct,
        vol_delta, rvol_20,
        stk[i], std3[i],
        hour_sin, hour_cos, dow_sin, dow_cos,
    ]
