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


def verify(onnx_bytes: bytes, model, n_features: int, up_index: int, trials: int = 256) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=(trials, n_features)).astype(np.float32)
    # a few realistic magnitudes: RSI-like 0..100, ADX 0..60
    x[:, :4] *= 0.002
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
    drift = float(np.max(np.abs(proba - ref)))
    print(f"    max |onnx - sklearn| over {trials} random vectors = {drift:.3e}")
    print(f"    P(class {up_index}) sample: {proba[:3, up_index].round(4).tolist()}")
    return drift


def export_one(path: str, expected_dim: int, expected_names: list[str],
               out_path: str, up_index: int, tolerance: float, force: bool) -> bool:
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
                return False
        elif list(np.asarray(classes).ravel()) != [0, 1]:
            print("    !! classes_ is not [0, 1]. Column order is not what the EA assumes.")
            print(f"       Set InpUpClassIndex so that column N == 'up'/'take'.")
            if not force:
                return False

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
            return False
        print("       --force given: exporting anyway. The result is not trustworthy.")

    onnx_model = convert(model, width)
    info = describe_graph(onnx_model)
    print(f"    graph inputs : {info['inputs']}")
    print(f"    graph outputs: {info['outputs']}")

    for o in info["outputs"]:
        if o["kind"] != "tensor":
            print(f"    !! output '{o['name']}' is a {o['kind']}, not a tensor. "
                  "MQL5 cannot read it -- zipmap=False did not take effect.")
            return False

    blob = onnx_model.SerializeToString()
    drift = verify(blob, model, width, up_index)
    if drift > tolerance:
        print(f"    !! drift {drift:.3e} exceeds tolerance {tolerance:.0e}; refusing to write")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(blob)
    print(f"    wrote {out_path} ({len(blob)/1024:.1f} KB)")

    sidecar = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source_pickle": os.path.abspath(path),
                "estimator": type(model).__name__,
                "input_width": width,
                "expected_width": expected_dim,
                "feature_order": expected_names,
                "up_class_index": up_index,
                "verify_max_drift": drift,
                "graph": info,
            },
            fh,
            indent=2,
        )
    print(f"    wrote {sidecar}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="primary model .pkl")
    ap.add_argument("--meta", required=True, help="meta model .pkl")
    ap.add_argument("--outdir", required=True, help="…/MQL5/Files/SniperAI")
    ap.add_argument("--up-index", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--force", action="store_true",
                    help="export even when the contract check fails (you have been warned)")
    args = ap.parse_args()

    ok_p = export_one(args.primary, FEATURE_DIM, FEATURE_NAMES,
                      os.path.join(args.outdir, "primary.onnx"),
                      args.up_index, args.tolerance, args.force)
    ok_m = export_one(args.meta, META_DIM, META_NAMES,
                      os.path.join(args.outdir, "meta.onnx"),
                      args.up_index, args.tolerance, args.force)

    print()
    if ok_p and ok_m:
        print("EXPORT OK -- set InpPrimaryModel / InpMetaModel to SniperAI\\primary.onnx "
              "and SniperAI\\meta.onnx")
        return 0
    print("EXPORT FAILED -- see the messages above. Do not attach the EA yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
