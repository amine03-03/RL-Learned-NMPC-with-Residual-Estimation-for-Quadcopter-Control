"""Experiment manager (§9.13, §9.14): sweeps, run directories, config capture.

Each run saves the tree of §9.14, and ``config.yaml`` records **git commit of
the controller and RDP code, PX4 version, Gazebo version, controller rate, RDP
rate, MPC horizon, RDP history length, disturbance parameters, random seed,
plant parameters, and the calibrated T_max in use**.  Without these the result
is not reproducible, so a run that cannot capture them says so in the file
rather than omitting the field.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _cmd(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:                                     # noqa: BLE001
        return "unavailable"


def capture_config(run_id, scenario, controller, seed, **extra):
    """Everything §9.14 requires.  Fields that cannot be read say so."""
    cfg = dict(
        run_id=run_id, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=_cmd(["git", "rev-parse", "HEAD"], cwd=ROOT),
        git_dirty=bool(_cmd(["git", "status", "--porcelain"], cwd=ROOT)),
        px4_version=os.environ.get("PX4_VERSION", _cmd(["px4", "--version"])),
        gazebo_version=os.environ.get("GZ_VERSION", _cmd(["gz", "sim", "--versions"])),
        ros_distro=os.environ.get("ROS_DISTRO", "unavailable"),
        host=platform.node(), python=platform.python_version(),
        controller=controller, scenario=scenario, seed=int(seed))
    cfg.update(extra)
    missing = [k for k, v in cfg.items() if v == "unavailable"]
    if missing:
        cfg["reproducibility_warning"] = (
            f"could not capture {missing}; this run is NOT fully reproducible "
            "and must not be quoted as such")
    return cfg


def run_dir(base, run_id):
    d = os.path.join(base, f"experiment_{run_id}")
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    return d


def save_config(d, cfg):
    try:
        import yaml
        with open(os.path.join(d, "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
    except Exception:                                     # noqa: BLE001
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2, default=str)
    return d


def save_metrics(d, metrics):
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# --------------------------------------------------------------------------- #
# §9.13 experiment definitions
# --------------------------------------------------------------------------- #
CONTROLLERS = {
    "C0": dict(name="Nominal ACMPC", d="zero", note="baseline"),
    "C1": dict(name="Oracle ACMPC", d="true", note="upper bound; NOT deployable"),
    "C2": dict(name="ACMPC + RDP", d="rdp", note="the proposed method"),
    "C3": dict(name="ACMPC + degraded RDP", d="rdp_degraded",
               note="robustness to imperfect prediction"),
}
INCUMBENTS = {"PID": dict(mode="pid"), "NMPC1": dict(mode="nmpc1")}


def plan_EA(scenarios=("S0", "S1", "S2", "S3", "S4", "S5", "S6"), seeds=(0, 1, 2)):
    """E-A scenario matrix: S0..S6 x C0..C3, three seeds, median and spread."""
    plan = [dict(exp="E-A", scenario=s, controller=c, seed=k)
            for s in scenarios for c in CONTROLLERS for k in seeds]
    # the realistic incumbents, on S0/S1 only, so the deployment result connects
    # to the study ledger
    plan += [dict(exp="E-A", scenario=s, controller=c, seed=k)
             for s in ("S0", "S1") for c in INCUMBENTS for k in seeds]
    return plan


def plan_EB(freqs=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), seeds=(0,)):
    """E-B frequency response, from S4."""
    return [dict(exp="E-B", scenario="S4", controller="C2", seed=k, freq_hz=f)
            for f in freqs for k in seeds] + \
           [dict(exp="E-B", scenario="S4", controller="C2", seed=k, chirp=True)
            for k in seeds]


def plan_EC(horizons=(10, 20, 30, 40), seeds=(0,)):
    """E-C MPC horizon interaction, under identical disturbances.

    The report must **mark which horizons violate (9.4)**, which is why the
    horizon is a planned axis rather than a fixed choice.
    """
    return [dict(exp="E-C", scenario="S3", controller=c, seed=k, horizon=N)
            for N in horizons for c in ("C0", "C1", "C2") for k in seeds]


def plan_ED(H=(16, 32, 64), seeds=(0,)):
    """E-D history length: accuracy and latency."""
    return [dict(exp="E-D", scenario="S3", controller="C2", seed=k, H=h)
            for h in H for k in seeds]


def full_plan():
    return plan_EA() + plan_EB() + plan_EC() + plan_ED()


def summarise(rows, by=("scenario", "controller"), value="pos_rmse"):
    """Median and spread across seeds -- never a single seed's number."""
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(list(by))[value]
    return g.agg(median="median", iqr=lambda s: float(s.quantile(.75) - s.quantile(.25)),
                 n="size").reset_index()


def main(argv=None):                                       # pragma: no cover
    """Print the planned experiment matrix and the captured config."""
    import json
    import sys
    plan = full_plan()
    print(f"planned runs: {len(plan)}")
    by = {}
    for r in plan:
        by[r["exp"]] = by.get(r["exp"], 0) + 1
    for k in sorted(by):
        print(f"  {k}: {by[k]}")
    print(json.dumps(capture_config("dry-run", "S0", "C0", 0), indent=2,
                     default=str))
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
