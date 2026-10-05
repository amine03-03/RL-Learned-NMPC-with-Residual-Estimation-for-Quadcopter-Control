"""S0..S6 of §9.7 -- deterministic, reproducible, seeds recorded in the run config.

Every scenario returns the **true** external wrench in [N (world), N.m (body)]
-- the RDP's target, relative to the nominal model -- and, where it applies, the
plant-parameter perturbation.  Parameter scenarios are expressed as the wrench
they cause, so that they can be both logged as ground truth and *applied* to the
Gazebo model (``manager_node`` -> ``ApplyLinkWrench``):

* S1 payload: a point mass ``m_p = (mass_scale - 1) m`` at the body offset
  ``cg_offset`` -> ``F = -m_p g e3`` (world), ``tau = d x R^T F`` (body).  At
  hover this is the ``-m_p g (d_y, -d_x, 0)`` of CORRECTIONS C-6.  Quasi-static:
  the payload's own inertial force ``-m_p a`` and ``inertia_scale`` are not
  realised (they are recorded in the run config only).
* S5 motor loss: rotor i delivers ``eta`` of its thrust, so the missing
  ``dT = (1-eta) K_T Omega_i^2`` (Omega_i from the ``actuator_motors`` command,
  A2 mapping) acts as ``F = -dT z_b`` at the rotor, ``tau = r_i x F`` plus the
  lost reaction torque ``+sigma_i k_m dT`` about z.

Every scenario acts during ``[t_on, t_off)`` only, so the onset and the removal
are both visible (§9.9 timeline).  The ground truth is
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
#: rotor model of the gz x500 (SDF + SIM_GZ_EC_*), asserted against the study
#: by check_glue.  Rotor order is PX4's motor order: 0 front-right CCW,
#: 1 back-left CCW, 2 front-left CW, 3 back-right CW (FLU, +1 = CCW from above).
K_T = 8.54858e-06                 # motorConstant [N s^2]
K_M = 0.016                       # momentConstant -- what the rotors produce
OM_MIN, OM_MAX = 150.0, 1000.0    # SIM_GZ_EC_MIN / MAX [rad/s]
R_ROTOR = np.array([[+0.174, -0.174, 0.06],
                    [-0.174, +0.174, 0.06],
                    [+0.174, +0.174, 0.06],
                    [-0.174, -0.174, 0.06]])
SIGMA = np.array([+1.0, +1.0, -1.0, -1.0])
E3 = np.array([0.0, 0.0, 1.0])

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
        """Parameter mismatch the ACMPC is NOT told about (it keeps nominal).

        Realised in Gazebo as the wrench of :func:`payload_wrench` /
        :func:`motor_loss_wrench`; ``inertia_scale`` is recorded only."""
        keys = ("mass_scale", "inertia_scale", "cg_offset", "motor_index",
                "motor_effectiveness")
        return {k: self.params[k] for k in keys if k in self.params}

    def active(self, t):
        return self.t_on <= t < self.t_off

    def wrench(self, t, R=None, pwm=None):
        """True external wrench at time t: (F_world (3) [N], tau_body (3) [N.m]).

        R   : body FLU -> world ENU rotation (identity if unknown)
        pwm : the four ``actuator_motors`` commands, needed by S5/S6
        """
        F = np.zeros(3)
        tau = np.zeros(3)
        if self.sid == "S0" or not self.active(t):
            return F, tau
        p = self.params
        R = np.eye(3) if R is None else np.asarray(R, float).reshape(3, 3)
        if self.sid in ("S1", "S6"):
            F_p, tau_p = payload_wrench(p["mass_scale"], p["cg_offset"], R)
            F, tau = F + F_p, tau + tau_p
        if self.sid in ("S5", "S6") and pwm is not None:
            F_m, tau_m = motor_loss_wrench(p["motor_index"],
                                           p["motor_effectiveness"], pwm, R)
            F, tau = F + F_m, tau + tau_m
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


def payload_wrench(mass_scale, cg_offset, R):
    """S1: gravity of a point payload at body offset d -> (F_world, tau_body)."""
    m_p = (float(mass_scale) - 1.0) * M_NOM
    F_w = -m_p * G * E3
    tau_b = np.cross(np.asarray(cg_offset, float), R.T @ F_w)
    return F_w, tau_b


def motor_loss_wrench(i, eta, pwm, R):
    """S5: the thrust and reaction torque rotor i no longer delivers."""
    c = float(np.clip(np.asarray(pwm, float)[int(i)], 0.0, 1.0))
    om = OM_MIN + c * (OM_MAX - OM_MIN)
    dT = (1.0 - float(eta)) * K_T * om ** 2
    F_b = -dT * E3
    tau_b = np.cross(R_ROTOR[int(i)], F_b) + SIGMA[int(i)] * K_M * dT * E3
    return R @ F_b, tau_b


def to_world(F_world, tau_body, R):
    """Gazebo's ``AddWorldWrench`` takes the torque in the world frame too."""
    return np.asarray(F_world, float), np.asarray(R, float) @ np.asarray(tau_body, float)


class PersistentWrench:
    """Drive gz-sim's ``ApplyLinkWrench`` persistent topic as a *setter*.

    ``ApplyLinkWrench`` appends every message on ``/world/<w>/wrench/persistent``
    to a list and applies the **sum** at every physics step; ``.../clear``
    removes them all at the next step (including any sent just before it).  The
    one-shot topic lasts a single 4 ms step, so at 50 Hz it would act 1/5 of the
    time.  Publishing the *difference* to the last commanded wrench keeps the
    applied sum equal to the latest wrench exactly.

    Increments are only exact if none is lost, so every ``resync_every`` calls
    the list is cleared and the full wrench re-sent on the following call: a
    lost message is corrected within one resync period, and the list stays
    bounded.  The cost is one tick at zero wrench per period (20 ms per 5 s at
    the defaults, 0.4 %).
    """

    def __init__(self, resync_every=250, eps=1e-9):
        self.applied = np.zeros(6)
        self.n = 0
        self.resync_every = int(resync_every)
        self.eps = eps
        self._cleared = False

    def step(self, F_world, tau_world):
        """-> ('clear', None) | ('add', delta (6,)) | (None, None)."""
        want = np.concatenate([F_world, tau_world]).astype(float)
        if self._cleared:                       # re-send after a compaction
            self._cleared = False
            self.applied = np.zeros(6)
        self.n += 1
        if self.n >= self.resync_every:
            self.n = 0
            self._cleared = True
            return "clear", None
        d = want - self.applied
        if np.max(np.abs(d)) <= self.eps:
            return None, None
        self.applied = want
        return "add", d

    def reset(self):
        self.applied = np.zeros(6)
        self.n = 0
        self._cleared = False


def sweep_S4(freqs=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), seed=0, **kw):
    """§9.7 S4: sweep the periodic disturbance frequency, plus a chirp variant."""
    out = [Scenario("S4", seed=seed, params=dict(frequency_hz=f, chirp=False), **kw)
           for f in freqs]
    out.append(Scenario("S4", seed=seed, params=dict(chirp=True), **kw))
    return out
