#!/usr/bin/env python3
"""
train_two_tier.py -- train the two-tier meta-labelling stack the V5 EA runs.

    python3 tools/train_two_tier.py --csv XAUUSD_M5.csv --outdir models/

WHAT THIS DOES DIFFERENTLY FROM A NAIVE PIPELINE
------------------------------------------------
1. TRIPLE-BARRIER LABELS, not "next bar up/down".
   The primary label is: starting at this bar, does price hit +k*ATR
   before -k*ATR within a horizon of H bars? That is the question the EA
   actually asks. "close[t+1] > close[t]" is a different question and a
   model trained on it will look excellent and lose money.

2. COSTS ARE IN THE LABEL.
   Every barrier is shifted by (spread + commission + slippage). On M5
   gold a 30-cent round-trip cost is a large fraction of a 1xATR move.
   A model trained cost-blind learns to scalp an edge that does not exist
   once your broker is paid.

3. PURGED, EMBARGOED WALK-FORWARD CV.
   Because a label at time t depends on bars up to t+H, an ordinary
   KFold leaks the future into the training fold. We purge the H bars
   around every fold boundary and embargo a further 1%. This is the
   single biggest reason ML trading backtests do not reproduce.

4. THE META MODEL IS TRAINED ON THE PRIMARY'S OUT-OF-FOLD CALLS.
   Training it on in-sample primary predictions teaches it that the
   primary is far better than it is, and it stops filtering anything.

5. NO SHUFFLING, NO RANDOM SPLIT, EVER.

Output: primary.pkl, meta.pkl, report.json in --outdir. Feed the two
pickles to tools/export_to_onnx.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sniper_features import (  # noqa: E402
    FEATURE_NAMES,
    META_NAMES,
    WARMUP_BARS,
    atr,
    build_features,
    build_meta_features,
)


# --------------------------------------------------------------------------
# labelling
# --------------------------------------------------------------------------
def triple_barrier(
    df: pd.DataFrame,
    horizon: int,
    up_mult: float,
    dn_mult: float,
    cost_price: float,
) -> pd.DataFrame:
    """
    For every bar t, walk forward up to `horizon` bars and record which
    barrier is touched first, for a LONG entered at close[t].

    Returns columns:
        y_dir   1 = the long barrier was hit first, 0 = the short one was
        hit     1 = a barrier was touched, 0 = the horizon expired
        ret     realised log return at the exit, net of cost
        t_exit  bar index of the exit (used for purging)
    """
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    a = atr(df["high"], df["low"], df["close"], 14).to_numpy(dtype="float64")
    n = len(c)

    y_dir = np.full(n, np.nan)
    hit = np.zeros(n, dtype="int8")
    ret = np.full(n, np.nan)
    t_exit = np.full(n, -1, dtype="int64")

    for t in range(n - 1):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        entry = c[t] + cost_price / 2.0          # you buy at the ask
        up = entry + up_mult * a[t]
        dn = entry - dn_mult * a[t]
        end = min(n - 1, t + horizon)

        exit_i = end
        outcome = None
        for k in range(t + 1, end + 1):
            hi_hit = h[k] >= up
            lo_hit = lo[k] <= dn
            if hi_hit and lo_hit:
                # both inside one bar: assume the adverse one first.
                outcome, exit_i = 0, k
                break
            if hi_hit:
                outcome, exit_i = 1, k
                break
            if lo_hit:
                outcome, exit_i = 0, k
                break

        exit_px = c[exit_i] - cost_price / 2.0    # you sell at the bid
        ret[t] = np.log(max(exit_px, 1e-9) / max(entry, 1e-9))
        t_exit[t] = exit_i
        if outcome is None:
            hit[t] = 0
            y_dir[t] = 1.0 if ret[t] > 0 else 0.0
        else:
            hit[t] = 1
            y_dir[t] = float(outcome)

    return pd.DataFrame(
        {"y_dir": y_dir, "hit": hit, "ret": ret, "t_exit": t_exit}, index=df.index
    )


# --------------------------------------------------------------------------
# purged walk-forward CV
# --------------------------------------------------------------------------
def purged_folds(n: int, n_splits: int, horizon: int, embargo_frac: float = 0.01):
    """Yield (train_idx, test_idx) with the label horizon purged and embargoed."""
    idx = np.arange(n)
    fold_size = n // n_splits
    embargo = int(n * embargo_frac)
    for k in range(n_splits):
        start = k * fold_size
        stop = n if k == n_splits - 1 else (k + 1) * fold_size
        test = idx[start:stop]
        left = idx[: max(0, start - horizon - embargo)]
        right = idx[min(n, stop + horizon + embargo):]
        train = np.concatenate([left, right])
        if len(train) < 500 or len(test) < 100:
            continue
        yield train, test


def make_model(kind: str, seed: int):
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=80, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1,
        )
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=5.0, random_state=seed, eval_metric="logloss",
            tree_method="hist",
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    # deliberately shallow. 40 features on M5 gold will memorise anything
    # you let them.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.04, max_depth=4,
        min_samples_leaf=80, l2_regularization=5.0,
        early_stopping=False, random_state=seed,
    )


def report_binary(y_true, p, threshold=0.5) -> dict:
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y_true)), "base_rate": float(np.mean(y_true))}
    try:
        out["auc"] = float(roc_auc_score(y_true, p))
    except ValueError:
        out["auc"] = float("nan")
    out["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    out["acc"] = float(accuracy_score(y_true, (p >= threshold).astype(int)))
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="OHLCV csv in BROKER server time")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--horizon", type=int, default=12, help="bars to the vertical barrier")
    ap.add_argument("--up-mult", type=float, default=1.5, help="upper barrier = n x ATR(14)")
    ap.add_argument("--dn-mult", type=float, default=1.5, help="lower barrier = n x ATR(14)")
    ap.add_argument("--cost", type=float, default=0.30,
                    help="round-trip cost in PRICE units (XAUUSD: spread+comm+slip, e.g. 0.30)")
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--model", choices=["hgb", "lightgbm", "xgboost"], default="hgb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="final untouched slice, evaluated once at the end")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]
    try:
        df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M:%S")
    except (ValueError, TypeError):
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"loaded {len(df)} bars: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    print("building features ...")
    X = build_features(df)
    print("labelling (triple barrier, cost-aware) ...")
    lab = triple_barrier(df, args.horizon, args.up_mult, args.dn_mult, args.cost)

    frame = pd.concat([X, lab], axis=1)
    frame = frame.iloc[WARMUP_BARS:]
    frame = frame[np.isfinite(frame[FEATURE_NAMES]).all(axis=1)]
    frame = frame.dropna(subset=["y_dir", "ret"])
    # the last `horizon` labels look into bars we may not have
    frame = frame.iloc[: len(frame) - args.horizon]
    frame = frame.reset_index(drop=True)
    print(f"usable rows: {len(frame)}  |  barrier-touch rate: {frame['hit'].mean():.3f}")
    print(f"long-barrier-first rate: {frame['y_dir'].mean():.3f}  (0.5 = balanced)")

    if frame["y_dir"].mean() > 0.62 or frame["y_dir"].mean() < 0.38:
        print("  !! labels are heavily one-sided. On a trending sample this is normal,")
        print("     but a model trained on it will inherit the bias and, like the V4")
        print("     logs, emit the same side forever. Consider a longer sample or")
        print("     symmetric barriers relative to a trend-detrended price.")

    n = len(frame)
    n_hold = int(n * args.holdout_frac)
    n_dev = n - n_hold
    dev = frame.iloc[:n_dev].reset_index(drop=True)
    hold = frame.iloc[n_dev:].reset_index(drop=True)
    print(f"development: {len(dev)}   final holdout: {len(hold)} (touched once, at the end)")

    Xd = dev[FEATURE_NAMES].to_numpy(dtype="float64")
    yd = dev["y_dir"].to_numpy(dtype="int64")

    # ---------------- tier 1, out-of-fold -------------------------------
    print("\n=== TIER 1 (direction) ===")
    oof = np.full(len(dev), np.nan)
    fold_reports = []
    for i, (tr, te) in enumerate(purged_folds(len(dev), args.splits, args.horizon)):
        m = make_model(args.model, args.seed + i)
        m.fit(Xd[tr], yd[tr])
        p = m.predict_proba(Xd[te])[:, 1]
        oof[te] = p
        r = report_binary(yd[te], p)
        fold_reports.append(r)
        print(f"  fold {i}: n={r['n']:6d} auc={r['auc']:.4f} logloss={r['logloss']:.4f} acc={r['acc']:.4f}")

    mask = np.isfinite(oof)
    primary_oof = report_binary(yd[mask], oof[mask])
    print(f"  OOF: auc={primary_oof['auc']:.4f} acc={primary_oof['acc']:.4f} "
          f"base={primary_oof['base_rate']:.4f}")
    if primary_oof["auc"] < 0.52:
        print("  !! OOF AUC below 0.52. There is no directional edge here. Everything")
        print("     downstream is curve fitting. Change the problem, not the model.")

    primary = make_model(args.model, args.seed)
    primary.fit(Xd, yd)

    # ---------------- tier 2, meta ---------------------------------------
    print("\n=== TIER 2 (meta / take-the-trade gate) ===")
    side = np.where(oof >= 0.5, 1.0, -1.0)
    # the meta label: taking the primary's call, did it pay AFTER costs?
    signed_ret = dev["ret"].to_numpy(dtype="float64") * side
    y_meta = (signed_ret > 0.0).astype("int64")

    Md = build_meta_features(dev[FEATURE_NAMES], np.nan_to_num(oof, nan=0.5), side)
    Md = Md[mask].to_numpy(dtype="float64")
    y_meta_m = y_meta[mask]
    print(f"  meta rows: {len(y_meta_m)}  profitable-after-cost rate: {y_meta_m.mean():.3f}")

    meta_oof = np.full(len(y_meta_m), np.nan)
    for i, (tr, te) in enumerate(purged_folds(len(y_meta_m), args.splits, args.horizon)):
        m = make_model(args.model, args.seed + 100 + i)
        m.fit(Md[tr], y_meta_m[tr])
        meta_oof[te] = m.predict_proba(Md[te])[:, 1]
    mmask = np.isfinite(meta_oof)
    meta_rep = report_binary(y_meta_m[mmask], meta_oof[mmask])
    print(f"  OOF: auc={meta_rep['auc']:.4f} acc={meta_rep['acc']:.4f} base={meta_rep['base_rate']:.4f}")

    meta = make_model(args.model, args.seed + 1)
    meta.fit(Md, y_meta_m)

    # ---------------- threshold sweep ------------------------------------
    print("\n=== THRESHOLD SWEEP (out-of-fold, net of cost) ===")
    sweep = []
    sr = signed_ret[mask][mmask]
    for thr in np.arange(0.30, 0.86, 0.05):
        take = meta_oof[mmask] >= thr
        k = int(take.sum())
        if k < 30:
            sweep.append({"threshold": round(float(thr), 2), "trades": k})
            continue
        r = sr[take]
        exp = float(r.mean())
        win = float((r > 0).mean())
        pf_up = float(r[r > 0].sum())
        pf_dn = float(-r[r < 0].sum())
        pf = pf_up / pf_dn if pf_dn > 0 else float("inf")
        sweep.append({"threshold": round(float(thr), 2), "trades": k,
                      "expectancy_log": exp, "winrate": win, "profit_factor": pf})
        print(f"  thr={thr:.2f}  trades={k:6d}  win={win:.3f}  PF={pf:6.3f}  E[r]={exp:+.6f}")
    print("\n  Pick the threshold with a positive expectancy AND enough trades to")
    print("  trust it. The V4 panel ran at 0.35, which on this table is usually")
    print("  'take almost everything'.")

    # ---------------- final holdout, evaluated once -----------------------
    print("\n=== FINAL HOLDOUT (never used for any decision above) ===")
    Xh = hold[FEATURE_NAMES].to_numpy(dtype="float64")
    yh = hold["y_dir"].to_numpy(dtype="int64")
    ph = primary.predict_proba(Xh)[:, 1]
    hold_primary = report_binary(yh, ph)
    side_h = np.where(ph >= 0.5, 1.0, -1.0)
    Mh = build_meta_features(hold[FEATURE_NAMES], ph, side_h).to_numpy(dtype="float64")
    pm = meta.predict_proba(Mh)[:, 1]
    sr_h = hold["ret"].to_numpy(dtype="float64") * side_h
    print(f"  primary: auc={hold_primary['auc']:.4f} acc={hold_primary['acc']:.4f}")
    hold_sweep = []
    for thr in (0.35, 0.50, 0.60, 0.70):
        take = pm >= thr
        k = int(take.sum())
        if k < 20:
            print(f"  thr={thr:.2f}  trades={k}  (too few to judge)")
            hold_sweep.append({"threshold": thr, "trades": k})
            continue
        r = sr_h[take]
        pf_dn = float(-r[r < 0].sum())
        pf = float(r[r > 0].sum()) / pf_dn if pf_dn > 0 else float("inf")
        print(f"  thr={thr:.2f}  trades={k:5d}  win={(r>0).mean():.3f}  "
              f"PF={pf:6.3f}  E[r]={r.mean():+.6f}")
        hold_sweep.append({"threshold": thr, "trades": k, "winrate": float((r > 0).mean()),
                           "profit_factor": pf, "expectancy_log": float(r.mean())})

    # ---------------- save ------------------------------------------------
    import joblib

    joblib.dump(primary, os.path.join(args.outdir, "primary.pkl"))
    joblib.dump(meta, os.path.join(args.outdir, "meta.pkl"))
    report = {
        "rows": int(n),
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "meta_names": META_NAMES,
        "primary_oof": primary_oof,
        "primary_folds": fold_reports,
        "meta_oof": meta_rep,
        "threshold_sweep_oof": sweep,
        "holdout_primary": hold_primary,
        "holdout_sweep": hold_sweep,
    }
    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)

    print(f"\nsaved primary.pkl, meta.pkl and report.json to {args.outdir}/")
    print("next: python3 tools/export_to_onnx.py --primary %s/primary.pkl --meta %s/meta.pkl "
          "--outdir <MQL5/Files/SniperAI>" % (args.outdir, args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
