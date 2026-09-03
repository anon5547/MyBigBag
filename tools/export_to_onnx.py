#!/usr/bin/env python3
"""
export_to_onnx.py -- turn a trained sklearn / XGBoost / LightGBM classifier
into an .onnx file that MQL5's native runtime can execute.

    python3 tools/export_to_onnx.py \
        --primary XAUUSD_M5_Trinity_Sniper.pkl \
        --meta    XAUUSD_M5_Bidirectional_Sniper.pkl \
        --outdir  "C:/Users/you/AppData/Roaming/MetaQuotes/Terminal/<id>/MQL5/Files/SniperAI"

Hard requirements this script ENFORCES, because MQL5 cannot recover from
any of them at runtime:

  * zipmap=False. The default sklearn->ONNX conversion emits the class
    probabilities as a sequence-of-maps. MQL5 cannot read that type and
    OnnxSetOutputShape will simply fail with a shape error.
  * float32 input, shape [1, N]. Not float64.
  * classes_ == [0, 1]. The EA reads column `--up-index` (default 1) as
    P(up) / P(take). Anything else silently inverts your signal.
  * the model's input width matches the contract (40 primary, 43 meta).

It then re-runs both the original model and the exported graph on random
vectors and refuses to write a file whose probabilities drift by more
than 1e-4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sniper_features import FEATURE_DIM, META_DIM, FEATURE_NAMES, META_NAMES  # noqa: E402



def _install_bool_attr_shim() -> None:
    """
    skl2onnx >= 1.17 hands HistGradientBoosting's `nodes_missing_value_tracks_true`
    to onnx.helper as a list of numpy bools, and onnx rejects booleans in an
    `ints` attribute ("Expected an int, got a boolean"). The fix is a one-line
    cast; upstream has not shipped it, so we do it here rather than telling you
    to pick a worse model.

    Idempotent, and a no-op on toolchains where conversion already works.
    """
    try:
        import skl2onnx.common._container as container
    except ImportError:
        return
    if getattr(container.ModelComponentContainer, "_sniper_bool_shim", False):
        return

    def coerce(v):
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, np.ndarray) and v.dtype == bool:
            return v.astype(np.int64)
        if isinstance(v, (list, tuple)) and any(isinstance(e, (bool, np.bool_)) for e in v):
            return [int(e) if isinstance(e, (bool, np.bool_)) else e for e in v]
        return v

    original = container.ModelComponentContainer.add_node

    def add_node(self, op_type, inputs, outputs, op_domain="", op_version=None,
                 name=None, **attrs):
        return original(self, op_type, inputs, outputs, op_domain=op_domain,
                        op_version=op_version, name=name,
                        **{k: coerce(v) for k, v in attrs.items()})

    container.ModelComponentContainer.add_node = add_node
    container.ModelComponentContainer._sniper_bool_shim = True


def load_pickle(path: str):
    import joblib

    try:
        return joblib.load(path)
    except Exception:
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)


def unwrap(obj):
    """Training scripts often pickle a dict; find the estimator inside."""
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for key in ("model", "clf", "estimator", "primary", "meta", "booster", "pipeline"):
            if key in obj and hasattr(obj[key], "predict_proba"):
                return obj[key]
        for v in obj.values():
            if hasattr(v, "predict_proba"):
                return v
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if hasattr(v, "predict_proba"):
                return v
    raise TypeError(
        "could not find an estimator with predict_proba inside the pickle "
        f"(got {type(obj).__name__})"
    )


def model_input_width(model) -> int | None:
    for attr in ("n_features_in_", "n_features_", "num_feature"):
        v = getattr(model, attr, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, (int, np.integer)) and v > 0:
            return int(v)
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        try:
            return int(len(booster().feature_names))
        except Exception:
            pass
    return None


def convert(model, n_features: int):
    """Return an ONNX ModelProto with a float32 [N, n_features] input."""
    _install_bool_attr_shim()
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("features", FloatTensorType([None, n_features]))]
    options = {id(model): {"zipmap": False}}

    name = type(model).__name__.lower()
    try:
        if "xgb" in name:
            from onnxmltools.convert import convert_xgboost

            return convert_xgboost(model, initial_types=initial_type, target_opset=15)
        if "lgbm" in name or "lightgbm" in name:
            from onnxmltools.convert import convert_lightgbm

            return convert_lightgbm(
                model, initial_types=initial_type, target_opset=15, zipmap=False
            )
    except ImportError as exc:
        raise SystemExit(
            f"{type(model).__name__} needs onnxmltools: pip install onnxmltools\n({exc})"
        )

    from skl2onnx import convert_sklearn

    return convert_sklearn(
        model, initial_types=initial_type, options=options, target_opset=15
    )


def describe_graph(onnx_model) -> dict:
    g = onnx_model.graph
    info = {"inputs": [], "outputs": []}
    for t in g.input:
        dims = [d.dim_value if d.dim_value else -1 for d in t.type.tensor_type.shape.dim]
        info["inputs"].append({"name": t.name, "elem_type": t.type.tensor_type.elem_type, "shape": dims})
    for t in g.output:
        kind = t.type.WhichOneof("value")
        if kind == "tensor_type":
            dims = [d.dim_value if d.dim_value else -1 for d in t.type.tensor_type.shape.dim]
            info["outputs"].append({"name": t.name, "kind": "tensor", "shape": dims})
        else:
            info["outputs"].append({"name": t.name, "kind": kind, "shape": None})
    return info


def build_verify_matrix(csv_path: str, n_features: int) -> np.ndarray | None:
    """
    Real feature vectors beat random ones by a mile here.

    A tree ensemble's error under float32 export is concentrated at the split
    thresholds. Random N(0,1) vectors sit far outside the training
    distribution, route to the same extreme leaves whatever the rounding, and
    report a drift of ~1e-7 while the real distribution drifts by 1e-1.
    Measured on 12,729 real gold bars: random said 1.3e-07 max, real said
    2.1e-01 max and 3.2% of gate decisions flipped.
    """
    if not csv_path:
        return None
    import pandas as pd

    from sniper_features import WARMUP_BARS, build_features

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    try:
        df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M:%S")
    except (ValueError, TypeError):
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    X = build_features(df).iloc[WARMUP_BARS:]
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) == 0:
        return None
    return X.to_numpy(dtype=np.float32)


def verify(onnx_bytes: bytes, model, n_features: int, up_index: int,
           x: np.ndarray | None, gate: float, max_flip_rate: float) -> tuple[bool, dict]:
    """Run both engines over the same vectors and compare where it counts."""
    import onnxruntime as ort

    synthetic = x is None
    if synthetic:
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, size=(2048, n_features)).astype(np.float32)
        x[:, :4] *= 0.002

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    ref = np.asarray(model.predict_proba(x.astype(np.float64)), dtype=np.float64)
    outs = sess.run(None, {in_name: x})
    proba = None
    for o in outs:
        arr = np.asarray(o)
        if arr.ndim == 2 and arr.shape[1] == ref.shape[1]:
            proba = arr.astype(np.float64)
            break
    if proba is None:
        raise SystemExit(
            "the exported graph has no [N, n_classes] float output. "
            "This is the zipmap problem -- MQL5 cannot read it."
        )

    d = np.abs(proba[:, up_index] - ref[:, up_index])
    flips = float(((proba[:, up_index] >= gate) != (ref[:, up_index] >= gate)).mean())
    stats = {
        "vectors": int(len(x)),
        "source": "random (NOT representative)" if synthetic else "real feature vectors",
        "median_drift": float(np.median(d)),
        "p99_drift": float(np.percentile(d, 99)),
        "max_drift": float(d.max()),
        "gate": gate,
        "decision_flip_rate": flips,
    }

    print(f"    verified on {stats['vectors']} {stats['source']}")
    print(f"      |onnx - sklearn|  median={stats['median_drift']:.2e}  "
          f"p99={stats['p99_drift']:.2e}  max={stats['max_drift']:.2e}")
    print(f"      decisions that flip at gate {gate:.2f}: {flips*100:.3f}%")

    if synthetic:
        print("    !! no --verify-csv given, so this was checked on random vectors,")
        print("       which do not exercise the split thresholds. Pass --verify-csv")
        print("       with the OHLCV file you trained on before trusting this number.")

    ok = flips <= max_flip_rate
    if not ok:
        print(f"    !! {flips*100:.3f}% of decisions disagree between the pickle and the")
        print(f"       exported graph (limit {max_flip_rate*100:.3f}%). The EA would take")
        print("       different trades than your backtest. Known cause:")
        print("       HistGradientBoostingClassifier bins features with float64 edges that")
        print("       do not survive the float32 ONNX conversion. Retrain with")
        print("       --model gb (GradientBoosting) or --model rf, both of which export")
        print("       exactly. Measured on real gold bars: HGB 1.75% flips, gb/rf 0.000%.")
    return ok, stats


def export_one(path: str, expected_dim: int, expected_names: list[str],
               out_path: str, up_index: int, force: bool,
               verify_x: np.ndarray | None, gate: float, max_flip_rate: float):
    """Convert and verify. Writes NOTHING - main() commits both files or neither."""
    print(f"\n=== {os.path.basename(path)} -> {os.path.basename(out_path)}")
    raw = load_pickle(path)
    model = unwrap(raw)
    print(f"    estimator: {type(model).__name__}")

    classes = getattr(model, "classes_", None)
    if classes is not None:
        print(f"    classes_: {list(classes)}")
        if len(classes) != 2:
            print("    !! not a binary classifier. The EA expects exactly two classes.")
            if not force:
                return False, None, None, None
        elif list(np.asarray(classes).ravel()) != [0, 1]:
            print("    !! classes_ is not [0, 1]. Column order is not what the EA assumes.")
            print(f"       Set InpUpClassIndex so that column N == 'up'/'take'.")
            if not force:
                return False, None, None, None

    width = model_input_width(model)
    if width is None:
        print(f"    ?? cannot determine input width; assuming {expected_dim}")
        width = expected_dim
    else:
        print(f"    input width: {width}")

    if width != expected_dim:
        print(f"    !! WIDTH MISMATCH: model wants {width} features, the V5 contract "
              f"defines {expected_dim}.")
        print("       This model was trained on a DIFFERENT feature set. Exporting it")
        print("       would feed it 40 numbers it has never seen, in an order it does")
        print("       not know. Retrain with tools/train_two_tier.py, or hand-write an")
        print("       adapter that reproduces the original feature order exactly.")
        print(f"       Contract order: {expected_names[:6]} ... ({expected_dim} total)")
        if not force:
            return False, None, None, None
        print("       --force given: exporting anyway. The result is not trustworthy.")

    onnx_model = convert(model, width)
    info = describe_graph(onnx_model)
    print(f"    graph inputs : {info['inputs']}")
    print(f"    graph outputs: {info['outputs']}")

    for o in info["outputs"]:
        if o["kind"] != "tensor":
            print(f"    !! output '{o['name']}' is a {o['kind']}, not a tensor. "
                  "MQL5 cannot read it -- zipmap=False did not take effect.")
            return False, None, None, None

    blob = onnx_model.SerializeToString()
    x = verify_x
    if x is not None and x.shape[1] != width:
        print(f"    ?? verification vectors are {x.shape[1]}-wide, model is {width}; "
              "falling back to random")
        x = None
    ok_verify, stats = verify(blob, model, width, up_index, x, gate, max_flip_rate)
    if not ok_verify and not force:
        print("    refusing to write. Fix the model, do not --force a graph that")
        print("    disagrees with the estimator you validated.")
        return False, None, None, None

    meta = {
        "source_pickle": os.path.abspath(path),
        "estimator": type(model).__name__,
        "input_width": width,
        "expected_width": expected_dim,
        "feature_order": expected_names,
        "up_class_index": up_index,
        "verification": stats,
        "graph": info,
    }
    print(f"    prepared {os.path.basename(out_path)} ({len(blob)/1024:.1f} KB)")
    return True, model, blob, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="primary model .pkl")
    ap.add_argument("--meta", required=True, help="meta model .pkl")
    ap.add_argument("--outdir", required=True, help="…/MQL5/Files/SniperAI")
    ap.add_argument("--up-index", type=int, default=1)
    ap.add_argument("--verify-csv", default=None,
                    help="OHLCV csv (ideally the one you trained on). Without it the "
                         "fidelity check runs on random vectors and means very little.")
    ap.add_argument("--gate", type=float, default=0.60,
                    help="the InpConfThreshold you intend to run, so the flip rate "
                         "is measured where your decisions actually happen")
    ap.add_argument("--max-flip-rate", type=float, default=0.001,
                    help="reject the export if more than this fraction of decisions "
                         "differ between the pickle and the graph (default 0.1%%)")
    ap.add_argument("--force", action="store_true",
                    help="export even when a check fails (you have been warned)")
    args = ap.parse_args()

    base_x = build_verify_matrix(args.verify_csv, FEATURE_DIM)
    if base_x is not None:
        print(f"verification set: {len(base_x)} real feature vectors from "
              f"{os.path.basename(args.verify_csv)}")
        if len(base_x) > 20000:
            base_x = base_x[-20000:]
    else:
        print("verification set: none supplied -- falling back to random vectors")

    # The meta model only ever sees vectors the primary produced, so build its
    # verification set the way the EA will at runtime. This uses the primary
    # PICKLE, not its export, so the meta check stays meaningful even when the
    # primary's own export is rejected.
    meta_x = None
    if base_x is not None:
        try:
            pm = unwrap(load_pickle(args.primary))
            if model_input_width(pm) in (None, FEATURE_DIM):
                p_up = pm.predict_proba(base_x.astype(np.float64))[:, args.up_index]
                side = np.where(p_up >= 0.5, 1.0, -1.0)
                meta_x = np.column_stack(
                    [base_x, p_up, side, np.abs(2.0 * p_up - 1.0)]
                ).astype(np.float32)
        except Exception as exc:                      # noqa: BLE001
            print(f"could not build meta verification vectors from the primary: {exc}")

    ok_p, _, blob_p, meta_p = export_one(
        args.primary, FEATURE_DIM, FEATURE_NAMES,
        os.path.join(args.outdir, "primary.onnx"),
        args.up_index, args.force, base_x, args.gate, args.max_flip_rate)

    ok_m, _, blob_m, meta_m = export_one(
        args.meta, META_DIM, META_NAMES,
        os.path.join(args.outdir, "meta.onnx"),
        args.up_index, args.force, meta_x, args.gate, args.max_flip_rate)

    print()
    if not (ok_p and ok_m):
        print("EXPORT FAILED -- nothing was written. The two tiers ship as a pair;")
        print("writing one of them would leave a mismatched model on disk.")
        return 1

    os.makedirs(os.path.abspath(args.outdir), exist_ok=True)
    for name, blob, sidecar in (("primary", blob_p, meta_p), ("meta", blob_m, meta_m)):
        onnx_path = os.path.join(args.outdir, f"{name}.onnx")
        with open(onnx_path, "wb") as fh:
            fh.write(blob)
        with open(os.path.splitext(onnx_path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)
        print(f"wrote {onnx_path}")

    print("\nEXPORT OK -- set InpPrimaryModel / InpMetaModel to "
          "SniperAI\\primary.onnx and SniperAI\\meta.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
