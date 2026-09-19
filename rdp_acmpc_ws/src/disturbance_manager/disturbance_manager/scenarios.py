"""S0..S7 of §9.7 -- deterministic, reproducible, seeds recorded in the run config.

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

SCENARIO_IDS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
PURPOSE = {
    "S0": "baseline stability and tracking",
    "S1": "residual-model learning (parameter mismatch)",
    "S2": "adaptation time T_adapt (bounded step)",
    "S3": "transient estimation (wind / gust)",
    "S4": "prediction bandwidth (periodic / chirp)",
    "S5": "actuator/model residual capture (motor degradation)",
    "S6": "final demonstration (combined stress)",
    "S7": "hover stabilisation under an asymmetric payload (CG offset)",
}

#: S7 default payload: 300 g slung 12 cm forward, 6 cm left and 4 cm below the
#: nominal CG.  Sized so the demonstration is *visible without being unflyable*:
#:
#:   m_p / m      = 14.53 %          F_z    = -2.9420 N
#:   tau_body     = (-0.1765, +0.3530, 0) N m,  |tau| = 0.3947 N m
#:   tau_y / (hover-constrained pitch authority 1.2137 N m) = 29.1 %
#:
#: The authority figure is the one that matters: at hover each rotor sits at
#: 5.061 N with 3.488 N of headroom before F_max, so the largest pitch moment
#: available WITHOUT dropping out of hover is 2 * 0.174 * 3.488 = 1.2137 N m.
#: A payload drawing 29 % of that leaves the three controllers room to differ.
#: One drawing 90 % would saturate all three and measure the allocator instead.
S7_PAYLOAD_M = 0.30
S7_PAYLOAD_R = (0.12, 0.06, -0.04)


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
        if self.sid == "S7":
            p.setdefault("payload_mass", S7_PAYLOAD_M)
            p.setdefault("payload_offset", list(S7_PAYLOAD_R))
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
                "motor_effectiveness", "payload_mass", "payload_offset")
        return {k: self.params[k] for k in keys if k in self.params}

    def active(self, t):
        return self.t_on <= t < self.t_off

    def wrench(self, t, q=None):
        """True external wrench at time t: (F_world (3) [N], tau_body (3) [N.m]).

        ``q`` is the current attitude (scalar-first, ENU/FLU).  It matters only
        for S7, where the payload weight is fixed in the **world** frame while
        the moment it makes about the CG is fixed in the **body** frame, so the
        two are related through the attitude and not by a constant.  Passing
        None assumes hover (R = I), which is what the analytic guide on the live
        panel uses and is exact to first order at the small tilts a hover holds.
        """
        F = np.zeros(3)
        tau = np.zeros(3)
        if self.sid == "S7":
            # A payload is bolted on; it is not switched on at t_on.  It is
            # present for the whole episode, which is the point of the
            # scenario: the controllers are compared on a STANDING error, not
            # on a transient they can ride out.
            return self.payload_wrench(q)
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

    def payload_wrench(self, q=None):
        """The residual wrench a rigid payload adds, relative to the NOMINAL model.

        For true mass ``m = m_n + m_p`` the model's leftover force (4.10) is
        ``F = -T z_b m_p / m``; in hover ``T z_b = m g e3`` so

            F_world = (0, 0, -m_p g),      tau_body = r_p x (R^T F_world)   (9.8)

        Both are exact statements about the *nominal-model residual*, which is
        what the RDP is trained to predict and what the bridge converts -- not
        about the forces on the airframe, which are larger and uninteresting
        here.
        """
        m_p = float(self.params["payload_mass"])
        r_p = np.asarray(self.params["payload_offset"], dtype=float)
        F = np.array([0.0, 0.0, -m_p * G])
        if q is None:
            F_body = F
        else:
            qq = np.asarray(q, float)
            qq = qq / np.linalg.norm(qq)
            w, x, y, z = qq
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            F_body = R.T @ F
        return F, np.cross(r_p, F_body)

    def payload_inertia_delta(self):
        """Parallel-axis contribution of a point payload, ``m_p (|r|^2 I - r r^T)``.

        Recorded in the run config so the inertia the plant actually has is
        written down.  The controller is NOT told it (that is the mismatch under
        test), and the estimator cannot be: it sees only state and actuators.
        """
        m_p = float(self.params["payload_mass"])
        r = np.asarray(self.params["payload_offset"], dtype=float)
        return m_p * (float(r @ r) * np.eye(3) - np.outer(r, r))

    def describe(self):
        return dict(sid=self.sid, purpose=PURPOSE[self.sid], seed=self.seed,
                    t_on=self.t_on, t_off=self.t_off, params=self.params)


def sweep_S4(freqs=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), seed=0, **kw):
    """§9.7 S4: sweep the periodic disturbance frequency, plus a chirp variant."""
    out = [Scenario("S4", seed=seed, params=dict(frequency_hz=f, chirp=False), **kw)
           for f in freqs]
    out.append(Scenario("S4", seed=seed, params=dict(chirp=True), **kw))
    return out
