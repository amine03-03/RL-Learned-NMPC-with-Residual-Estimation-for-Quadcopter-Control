"""S0..S6 of §9.7 -- deterministic, reproducible, seeds recorded in the run config.

Every scenario returns the **true** external wrench in [N (world), N.m (body)]
and, where it applies, the plant-parameter perturbation.  The ground truth is
published on ``/disturbance/ground_truth`` for the evaluator and the logger
**only**; §9.1 forbids it reaching the online controller, and the RDP's window
is built from a ring buffer this module cannot write to.

Pure NumPy; importable with no ROS on the path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

M_NOM = 2.0643076923076924        # asserted equal to the study by check_glue
G = 9.8066
ARM = 0.174

SCENARIO_IDS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
PURPOSE = {
    "S0": "baseline stability and tracking",
    "S1": "residual-model learning (parameter mismatch)",
    "S2": "adaptation time T_adapt (bounded step)",
    "S3": "transient estimation (wind / gust)",
    "S4": "prediction bandwidth (periodic / chirp)",
    "S5": "actuator/model residual capture (motor degradation)",
    "S6": "final demonstration (combined stress)",
}


@dataclass
class Scenario:
    """A disturbance scenario evaluated at a wall time within the episode."""

    sid: str
    seed: int = 0
    t_on: float = 5.0
    t_off: float = 12.0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.sid not in SCENARIO_IDS:
            raise ValueError(f"unknown scenario {self.sid!r}")
        self.rng = np.random.default_rng(self.seed)
        self._draw()

    # -- parameter mismatch (applies for the whole episode) ------------------ #
    def _draw(self):
        r = self.rng
        p = dict(self.params)
        if self.sid in ("S1", "S6"):
            p.setdefault("mass_scale", float(r.uniform(1.10, 1.20)))
            p.setdefault("inertia_scale", float(r.uniform(0.85, 1.15)))
            p.setdefault("cg_offset", (r.uniform(0.02, 0.05)
                                       * r.choice([-1.0, 1.0], size=3)
                                       * np.array([1.0, 1.0, 0.4])).tolist())
        if self.sid in ("S5", "S6"):
            p.setdefault("motor_index", int(r.integers(0, 4)))
            p.setdefault("motor_effectiveness", 0.90)
        if self.sid in ("S2", "S6"):
            p.setdefault("step_force", [float(0.12 * M_NOM * G), 0.0, 0.0])
            p.setdefault("step_moment", [0.0, float(0.04 * M_NOM * G * ARM), 0.0])
        if self.sid in ("S3", "S6"):
            p.setdefault("gust_peak", float(0.15 * M_NOM * G))
            p.setdefault("gust_azimuth", float(r.uniform(0, 2 * np.pi)))
        if self.sid == "S4":
            p.setdefault("amplitude", float(0.10 * M_NOM * G))
            p.setdefault("frequency_hz", 0.5)
            p.setdefault("chirp", False)
            p.setdefault("chirp_f0", 0.1)
            p.setdefault("chirp_f1", 4.0)
        self.params = p

    def plant_overrides(self):
        """Parameter mismatch the ACMPC is NOT told about (it keeps nominal)."""
        keys = ("mass_scale", "inertia_scale", "cg_offset", "motor_index",
                "motor_effectiveness")
        return {k: self.params[k] for k in keys if k in self.params}

    def active(self, t):
        return self.t_on <= t < self.t_off

    def wrench(self, t):
        """True external wrench at time t: (F_world (3) [N], tau_body (3) [N.m])."""
        F = np.zeros(3)
        tau = np.zeros(3)
        if self.sid == "S0" or not self.active(t):
            return F, tau
        p = self.params
        if self.sid in ("S2", "S6"):
            F = F + np.asarray(p["step_force"], dtype=float)
            tau = tau + np.asarray(p["step_moment"], dtype=float)
        if self.sid in ("S3", "S6"):
            # finite-duration time-varying force, smooth at both ends so the
            # onset is a transient to estimate rather than a discontinuity
            span = max(self.t_off - self.t_on, 1e-6)
            u = (t - self.t_on) / span
            env = np.sin(np.pi * u) ** 2
            az = p["gust_azimuth"]
            mag = p["gust_peak"] * env * (1.0 + 0.3 * np.sin(2 * np.pi * 0.7 * t))
            F = F + mag * np.array([np.cos(az), np.sin(az), 0.0])
        if self.sid == "S4":
            if p["chirp"]:
                span = max(self.t_off - self.t_on, 1e-6)
                u = (t - self.t_on) / span
                f = p["chirp_f0"] * (p["chirp_f1"] / p["chirp_f0"]) ** u
                ph = 2 * np.pi * p["chirp_f0"] * span * (
                    (p["chirp_f1"] / p["chirp_f0"]) ** u - 1.0) \
                    / np.log(p["chirp_f1"] / p["chirp_f0"])
            else:
                f = p["frequency_hz"]
                ph = 2 * np.pi * f * (t - self.t_on)
            F = F + np.array([p["amplitude"] * np.sin(ph), 0.0, 0.0])
        return F, tau

    def describe(self):
        return dict(sid=self.sid, purpose=PURPOSE[self.sid], seed=self.seed,
                    t_on=self.t_on, t_off=self.t_off, params=self.params)


def sweep_S4(freqs=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), seed=0, **kw):
    """§9.7 S4: sweep the periodic disturbance frequency, plus a chirp variant."""
    out = [Scenario("S4", seed=seed, params=dict(frequency_hz=f, chirp=False), **kw)
           for f in freqs]
    out.append(Scenario("S4", seed=seed, params=dict(chirp=True), **kw))
    return out
