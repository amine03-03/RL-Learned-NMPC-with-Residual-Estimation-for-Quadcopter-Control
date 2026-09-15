"""export_estimator.py -- the export bridge of §9.5.

    python export_estimator.py --arch GRU --out ../rdp_acmpc_ws/models/rdp_gru.npz

The study writes JAX pytrees; the node must not depend on JAX or PyTorch.  This
writes a framework-neutral .npz plus a JSON metadata blob, and **asserts parity**
between the JAX predictor and the pure-NumPy ``rdp_infer`` over 64 random
windows at 1e-5.  Parity failure fails the export: a deployment model that has
silently diverged from the trained one is the single most expensive bug
available here, because nothing downstream would notice.

Choose the encoder on **latency**, not R^2 (§9.5): a predictor at 137 ms with
R^2 = 0.88 is not deployable; one at 2 ms with R^2 = 0.82 is.  ``--bench``
measures ``rdp_infer`` p95 on this machine and records it in the metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_core_jax as A       # noqa: E402
import x500_core_jax as X           # noqa: E402


def flatten_params(p):
    """JAX pytree -> flat {name: ndarray}, plus the extra metadata the NumPy
    forward pass needs to rebuild the topology."""
    W, meta = {}, {}
    kind = p["kind"]
    if kind in ("GRU", "LSTM"):
        for lyr in ("l1", "l2"):
            for k in ("Wi", "bi", "Wh", "bh"):
                W[f"{lyr}.{k}"] = np.asarray(p[lyr][k], dtype=np.float64)
        meta["nh"] = [int(p["l1"]["Wh"].shape[0]), int(p["l2"]["Wh"].shape[0])]
    elif kind == "TCN":
        for i, blk in enumerate(p["enc"]["blocks"]):
            W[f"b{i}.c1.W"] = np.asarray(blk["c1"]["W"], dtype=np.float64)
            W[f"b{i}.c1.b"] = np.asarray(blk["c1"]["b"], dtype=np.float64)
            W[f"b{i}.c2.W"] = np.asarray(blk["c2"]["W"], dtype=np.float64)
            W[f"b{i}.c2.b"] = np.asarray(blk["c2"]["b"], dtype=np.float64)
            W[f"b{i}.res.W"] = np.asarray(blk["res"]["W"], dtype=np.float64)
            W[f"b{i}.res.b"] = np.asarray(blk["res"]["b"], dtype=np.float64)
        meta["n_block"] = len(p["enc"]["blocks"])
    elif kind == "CNN":
        for i, lyr in enumerate(p["enc"]["layers"]):
            W[f"c{i}.W"] = np.asarray(lyr["W"], dtype=np.float64)
            W[f"c{i}.b"] = np.asarray(lyr["b"], dtype=np.float64)
        meta["n_layer"] = len(p["enc"]["layers"])
    else:
        raise ValueError(kind)
    W["head.W"] = np.asarray(p["head"]["W"], dtype=np.float64)
    W["head.b"] = np.asarray(p["head"]["b"], dtype=np.float64)
    return W, meta


def export(entry, out_path, bench=True, n_parity=64, tol=1e-5, seed=0):
    import jax.numpy as jnp
    import rdp_infer

    params, scales, H = A.rebuild_rdp(entry)
    W, extra = flatten_params(params)
    meta = dict(kind=entry["kind"], kw=entry.get("kw", {}), H=int(H),
                frame_dim=A.FRAME_DIM,
                mu=np.asarray(scales["mu"]).tolist(),
                sd=np.asarray(scales["sd"]).tolist(),
                out_sd=np.asarray(scales["out_sd"]).tolist(),
                channels=list(A.CHANNEL_NAMES), units=list(A.CHANNEL_UNITS),
                r2=entry.get("r2"), exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime()))
    meta.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(out_path, __meta__=json.dumps(meta), **W)

    rdp = rdp_infer.load(out_path)
    rng = np.random.default_rng(seed)
    win = rng.normal(size=(n_parity, H, A.FRAME_DIM)) * np.asarray(scales["sd"]) \
        + np.asarray(scales["mu"])
    y_np = rdp.predict(win)
    y_jx = np.asarray(A.rdp_apply(params, jnp.asarray(win), scales["mu"], scales["sd"])
                      * scales["out_sd"])
    err = float(np.abs(y_np - y_jx).max())
    ok = err < tol
    print(f"  parity (T-17): max |numpy - jax| = {err:.3e} over {n_parity} "
          f"random windows  [{'PASS' if ok else 'FAIL'}, tol {tol:g}]")
    if not ok:
        os.remove(out_path)
        raise SystemExit(
            f"EXPORT FAILED: parity {err:.3e} >= {tol:g}.  The deployment model "
            f"does not reproduce the trained one; refusing to write {out_path}.")

    lat = None
    if bench:
        one = win[:1]
        rdp.predict(one)
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            rdp.predict(one)
            ts.append(1e3 * (time.perf_counter() - t0))
        lat = {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95)),
               "max": float(np.max(ts))}
        meta["latency_ms_numpy"] = lat
        meta["latency_host"] = os.uname().nodename
        np.savez(out_path, __meta__=json.dumps(meta), **W)
        verdict = "admissible" if lat["p95"] < 20.0 else "NOT ADMISSIBLE"
        print(f"  rdp_infer single-window latency on {meta['latency_host']}: "
              f"median {lat['median']:.3f} ms, p95 {lat['p95']:.3f} ms "
              f"-> {verdict} against the 20 ms period")
    print(f"  wrote {out_path}  ({sum(v.size for v in W.values())} parameters)")
    return {"path": out_path, "parity": err, "latency": lat, "meta": meta}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default=None,
                    help="encoder name; default is the checkpoint's __selected__")
    ap.add_argument("--ckpt", default=None,
                    help="default artifacts/acmpc_adaptive/estimators/all.pkl")
    ap.add_argument("--out", default="../rdp_acmpc_ws/models/rdp_gru.npz")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(argv)

    ck = (X.load_ckpt("acmpc_adaptive", "estimators", "all.pkl") if a.ckpt is None
          else __import__("pickle").load(open(a.ckpt, "rb")))
    if ck is None:
        raise SystemExit(
            "no estimator checkpoint found at "
            "artifacts/acmpc_adaptive/estimators/all.pkl -- run Notebook 5 first.")
    name = a.arch or ck.get("__selected__")
    if name not in ck:
        raise SystemExit(f"architecture {name!r} not in checkpoint; "
                         f"available: {sorted(k for k in ck if not k.startswith('__'))}")
    print(f"exporting {name} (selected: {ck.get('__selected__')})")
    export(ck[name], a.out, bench=not a.no_bench, tol=a.tol)


if __name__ == "__main__":
    main()
