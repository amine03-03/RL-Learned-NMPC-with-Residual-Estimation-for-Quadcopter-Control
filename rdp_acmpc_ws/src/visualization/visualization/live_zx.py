"""Real-time Z-X plane for the three controllers, plus the adaptive estimates.

    ros2 run visualization live_zx                 # live, during a demo
    ros2 run visualization live_zx --replay runs/<session>   # from saved traces

The demonstration is a hover hold under an asymmetric payload (S7), flown three
times: ``nmpc1``, ``acmpc``, ``acmpc_adaptive``.  Only one can fly a given SITL
at a time, so the traces **accumulate**: each run is appended to
``zx_traces.npz`` in the session directory and reloaded at start-up, and the
figure therefore shows the finished controllers as static curves beside the one
currently flying.  That is what makes a single live window a comparison rather
than three windows nobody looks at together.

Layout::

    +---------------------------+------------------+
    |                           |  m_hat(t)        |   <- adaptive only
    |   Z-X plane               +------------------+
    |   (altitude vs x)         |  tau_hat(t)      |   <- adaptive only
    +---------------------------+------------------+
    |   z(t), all controllers, with the hold setpoint                |
    +----------------------------------------------------------------+

The Z-X plane is the headline because the payload's signature is a *coupled*
error: the standing pitch moment pushes the vehicle along +x while the extra
14.5 % of weight pulls z down, so the deviation is a diagonal excursion from
the setpoint and not an altitude drop.  Plotting z against t alone would hide
half of it; plotting z against x shows the whole thing in one curve.
"""
from __future__ import annotations

import os

import numpy as np

from . import estimates as EST

#: Order is the order they are flown and the order of the legend.
CONTROLLERS = ("nmpc1", "acmpc", "acmpc_adaptive")
LABELS = {"nmpc1": "NMPC, $N_p = 1$",
          "acmpc": "AC-MPC (learned cost)",
          "acmpc_adaptive": "adaptive AC-MPC (RDP in the model)"}
COLORS = {"nmpc1": "#4C72B0", "acmpc": "#DD8452", "acmpc_adaptive": "#55A868"}
STYLES = {"nmpc1": "-", "acmpc": "--", "acmpc_adaptive": "-"}

#: PX4 maps the normalised actuator command onto [OM_MIN, OM_MAX] (A2), so
#: thrust is NOT proportional to c^2.  Duplicated from check_ctbr and asserted
#: equal to it by check_demo.
K_T, OM_MIN, OM_MAX = 8.54858e-06, 150.0, 1000.0


def thrust_from_actuator_motors(c):
    """Total thrust [N] from the four normalised ``actuator_motors`` controls.

    ``Omega = OM_MIN + c (OM_MAX - OM_MIN)``, then ``F = sum K_T Omega^2``.
    Feeding the collective command straight in instead would overstate thrust
    at low command by the idle floor's worth -- 0.77 N, 3.8 % of weight.
    """
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    om = OM_MIN + c * (OM_MAX - OM_MIN)
    return float(K_T * np.sum(om ** 2))


class Trace:
    """One controller's history.  Plain lists; a hover demo is ~1500 samples."""

    FIELDS = ("t", "x", "y", "z", "qw", "qx", "qy", "qz",
              "Fx", "Fy", "Fz", "Mx", "My", "Mz", "T")

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, [])

    def __len__(self):
        return len(self.t)

    def push_pose(self, t, p, q=None):
        """``q`` (scalar-first ENU/FLU) is optional; without it the mass panel
        falls back to the hover form, which is what it will then report."""
        self.t.append(float(t))
        self.x.append(float(p[0])); self.y.append(float(p[1])); self.z.append(float(p[2]))
        qq = (np.asarray(q, float).reshape(4) if q is not None
              else np.array([np.nan] * 4))
        for f, v in zip(("qw", "qx", "qy", "qz"), qq):
            getattr(self, f).append(float(v))
        # pad the estimate channels so every column stays the same length even
        # when the estimator is slower than the pose, or absent entirely
        for f in ("Fx", "Fy", "Fz", "Mx", "My", "Mz", "T"):
            g = getattr(self, f)
            g.append(g[-1] if g else (np.nan if f == "T" else 0.0))

    def set_estimate(self, d6, thrust=None):
        """Overwrite the most recent sample's estimate channels."""
        if not self.t:
            return
        d = np.asarray(d6, float).reshape(6)
        for f, v in zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), d):
            getattr(self, f)[-1] = float(v)
        if thrust is not None:
            self.T[-1] = float(thrust)

    def arrays(self):
        return {f: np.asarray(getattr(self, f), float) for f in self.FIELDS}

    def force(self):
        a = self.arrays()
        return np.stack([a["Fx"], a["Fy"], a["Fz"]], -1)

    def moment(self):
        a = self.arrays()
        return np.stack([a["Mx"], a["My"], a["Mz"]], -1)

    def attitude(self):
        a = self.arrays()
        return np.stack([a["qw"], a["qx"], a["qy"], a["qz"]], -1)

    def mass(self):
        """-> (m_hat, m_p_hat, exact) -- the exact inversion of (9.9) where the
        thrust and the attitude are both recorded, the hover form (9.10) where
        they are not.  ``exact`` is reported on the panel, not assumed."""
        a = self.arrays()
        T, q = a["T"], self.attitude()
        exact = bool(np.isfinite(T).mean() > 0.5 and np.isfinite(q).all(-1).mean() > 0.5)
        if not exact:
            return (*EST.mass_estimate(self.force()), False)
        qf = np.where(np.isfinite(q), q, np.array([1.0, 0.0, 0.0, 0.0]))
        m, m_p = EST.mass_estimate(self.force(), np.nan_to_num(T, nan=0.0), qf)
        return m, m_p, True


class DemoRecorder:
    """Traces for all three controllers, with save/load so runs accumulate."""

    def __init__(self, setpoint=(0.0, 0.0, 1.5), truth=None):
        self.traces = {c: Trace() for c in CONTROLLERS}
        self.setpoint = np.asarray(setpoint, float)
        self.truth = truth                       # from estimates.payload_truth
        self.active = None
        self.t0 = {}

    def select(self, name):
        """Switch the active controller.  Unknown names are kept, not dropped:
        a typo in a launch argument should show up as an extra legend entry, not
        as a silently empty plot."""
        if name not in self.traces:
            self.traces[name] = Trace()
        self.active = name
        return name

    def push_pose(self, t, p, q=None, name=None):
        n = name or self.active
        if n is None:
            return
        self.t0.setdefault(n, float(t))
        self.traces[n].push_pose(float(t) - self.t0[n], p, q)

    def set_estimate(self, d6, thrust=None, name=None):
        n = name or self.active
        if n is not None:
            self.traces[n].set_estimate(d6, thrust)

    # -- persistence -------------------------------------------------------- #
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        out = {}
        for name, tr in self.traces.items():
            if len(tr):
                for f, v in tr.arrays().items():
                    out[f"{name}/{f}"] = v
        out["__setpoint__"] = self.setpoint
        np.savez_compressed(path, **out)
        return path

    def load(self, path, keep_active=True):
        """Merge saved traces in.  The active controller is NOT overwritten --
        reloading mid-flight must not discard the samples taken since start-up."""
        if not os.path.exists(path):
            return self
        z = np.load(path, allow_pickle=False)
        names = sorted({k.split("/")[0] for k in z.files if "/" in k})
        for name in names:
            if keep_active and name == self.active:
                continue
            tr = Trace()
            for f in Trace.FIELDS:
                key = f"{name}/{f}"
                setattr(tr, f, list(z[key]) if key in z.files else [])
            n = len(tr.t)
            for f in Trace.FIELDS:                # tolerate an older or ragged file
                v = list(getattr(tr, f))
                if len(v) != n:
                    setattr(tr, f, (v + [np.nan] * (n - len(v)))[:n])
            self.traces[name] = tr
        if "__setpoint__" in z.files:
            self.setpoint = z["__setpoint__"]
        return self

    # -- the numbers under the curves --------------------------------------- #
    def metrics(self):
        """Per controller: the standing errors the payload produces.

        ``z_bias`` and ``x_bias`` are means over the last 5 s, not RMS over the
        whole run: the payload is bolted on for the entire episode, so what
        separates the controllers is the *offset they settle to*, and an RMS
        taken from t = 0 buries it under the common take-off transient.
        """
        out = {}
        for name, tr in self.traces.items():
            if not len(tr):
                continue
            a = tr.arrays()
            t = a["t"]
            tail = t >= max(t.max() - 5.0, 0.0)
            if tail.sum() < 2:
                tail = np.ones_like(t, bool)
            e = np.stack([a["x"], a["y"], a["z"]], -1) - self.setpoint
            out[name] = dict(
                n=int(t.size), duration_s=float(t.max()),
                z_bias=float(e[tail, 2].mean()), x_bias=float(e[tail, 0].mean()),
                z_rms=float(np.sqrt((e[tail, 2] ** 2).mean())),
                p_rms=float(np.sqrt((e[tail] ** 2).sum(-1).mean())),
                p_max=float(np.linalg.norm(e, axis=-1).max()))
        return out


# --------------------------------------------------------------------------- #
#  rendering
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "legend.frameon": False})


def make_figure(agg=False):
    """``agg=True`` builds the figure off pyplot entirely.

    ``matplotlib.use("Agg")`` is global: calling it inside the periodic save
    would switch the backend of the interactive window the node is drawing to.
    A bare ``Figure`` with an Agg canvas renders and saves without touching
    pyplot's state at all, so the live window survives every save.
    """
    from matplotlib.gridspec import GridSpec
    _style()
    if agg:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(13.0, 7.2))
        FigureCanvasAgg(fig)
    else:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13.0, 7.2))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 0.62],
                  hspace=0.38, wspace=0.28,
                  left=0.06, right=0.985, top=0.90, bottom=0.08)
    ax = dict(zx=fig.add_subplot(gs[0:2, 0:2]),
              mass=fig.add_subplot(gs[0, 2]),
              mom=fig.add_subplot(gs[1, 2]),
              zt=fig.add_subplot(gs[2, :]))
    return fig, ax


def draw(fig, ax, rec, adaptive="acmpc_adaptive", title=None):
    """Redraw the whole panel from ``rec``.  Called at a few Hz; fast enough."""
    sp = rec.setpoint
    tr_truth = rec.truth

    # -- Z-X plane ---------------------------------------------------------- #
    a = ax["zx"]; a.clear(); a.grid(True, alpha=0.3)
    for name in list(CONTROLLERS) + [k for k in rec.traces if k not in CONTROLLERS]:
        tr = rec.traces.get(name)
        if not tr or not len(tr):
            continue
        d = tr.arrays()
        live = name == rec.active
        a.plot(d["x"], d["z"], STYLES.get(name, "-"), color=COLORS.get(name, "0.4"),
               lw=2.0 if live else 1.3, alpha=1.0 if live else 0.75,
               label=LABELS.get(name, name) + (" (flying)" if live else ""))
        a.plot(d["x"][0], d["z"][0], "o", ms=4, mfc="none",
               color=COLORS.get(name, "0.4"))
        a.plot(d["x"][-1], d["z"][-1], "o", ms=6 if live else 4,
               color=COLORS.get(name, "0.4"))
    a.plot(sp[0], sp[2], "*", ms=15, color="k", zorder=5, label="hold setpoint")
    a.axhline(sp[2], color="0.6", lw=0.8, ls=":")
    a.axvline(sp[0], color="0.6", lw=0.8, ls=":")
    a.set_xlabel("$x$ [m]  (payload leans the vehicle this way)")
    a.set_ylabel("$z$ [m]  (altitude)")
    a.set_title("Z-X plane -- hover hold under the asymmetric payload", fontsize=9)
    a.legend(fontsize=7, loc="best")
    a.set_aspect("equal", adjustable="datalim")

    # -- mass estimate ------------------------------------------------------ #
    a = ax["mass"]; a.clear(); a.grid(True, alpha=0.3)
    tr = rec.traces.get(adaptive)
    a.axhline(EST.M_NOM, color="0.5", lw=1.0, ls=":", label="$m_\\mathrm{nom}$")
    if tr_truth:
        a.axhline(tr_truth["m_total"], color="C3", lw=1.2, ls="--",
                  label="$m_\\mathrm{nom} + m_p$ (true)")
    if tr and len(tr):
        d = tr.arrays()
        m_hat, m_p, use_T = tr.mass()
        a.plot(d["t"], m_hat, "-", color=COLORS[adaptive], lw=1.4,
               label="$\\hat m$  " + ("(9.9, thrust-exact)" if use_T else "(9.10, hover)"))
        if tr_truth:
            ts = EST.settling_time(d["t"], m_hat, tr_truth["m_total"], 0.05, 0.5)
            if np.isfinite(ts):
                a.axvline(ts, color="0.4", lw=0.9, ls="-.")
                a.annotate(f"$T_\\mathrm{{adapt}}$ = {ts:.1f} s", (ts, 0.5),
                           xycoords=("data", "axes fraction"), fontsize=6.5,
                           xytext=(3, 0), textcoords="offset points")
        mp_tail = (np.nanmean(m_p[-50:]) if np.isfinite(m_p[-50:]).any()
                   else float("nan"))
        a.text(0.02, 0.97,
               f"$\\hat m_p$ = {mp_tail:+.3f} kg"
               + ("" if use_T else "   (no thrust/attitude: hover form)"),
               transform=a.transAxes, ha="left", va="top", fontsize=6.5)
    else:
        a.text(0.5, 0.5, "adaptive AC-MPC not flown yet", transform=a.transAxes,
               ha="center", va="center", fontsize=8, color="0.5")
    a.set_ylabel("mass [kg]"); a.set_xlabel("t [s]")
    a.set_title("mass estimate, adaptive AC-MPC", fontsize=9)
    a.margins(y=0.22)
    a.legend(fontsize=6.5, loc="lower right")

    # -- moment estimate ---------------------------------------------------- #
    a = ax["mom"]; a.clear(); a.grid(True, alpha=0.3)
    if tr and len(tr):
        d = tr.arrays()
        tau = tr.moment()
        for i, (lb, c) in enumerate((("$\\hat\\tau_x$", "C0"),
                                     ("$\\hat\\tau_y$", "C1"),
                                     ("$\\hat\\tau_z$", "C2"))):
            a.plot(d["t"], tau[:, i], "-", color=c, lw=1.3, label=lb)
            if tr_truth:
                a.axhline(tr_truth["tau"][i], color=c, lw=1.0, ls="--", alpha=0.6)
        _, m_p, _ = tr.mass()
        rx, ry = EST.cg_offset_estimate(tau, m_p)
        if np.isfinite(rx[-50:]).any():
            a.text(0.02, 0.03,
                   f"$\\hat r$ = ({np.nanmean(rx[-50:]):+.3f}, "
                   f"{np.nanmean(ry[-50:]):+.3f}) m"
                   + (f"   true ({tr_truth['r_x']:+.3f}, {tr_truth['r_y']:+.3f})"
                      if tr_truth else ""),
                   transform=a.transAxes, ha="left", va="bottom", fontsize=6.5)
    else:
        a.text(0.5, 0.5, "adaptive AC-MPC not flown yet", transform=a.transAxes,
               ha="center", va="center", fontsize=8, color="0.5")
    a.set_ylabel(r"moment [N m]"); a.set_xlabel("t [s]")
    a.set_title("moment estimate (dashed = true), adaptive AC-MPC", fontsize=9)
    a.margins(y=0.30)
    if a.get_legend_handles_labels()[0]:          # empty before the first flight
        a.legend(fontsize=6.5, ncol=3, loc="upper center")

    # -- altitude against time, all three ----------------------------------- #
    a = ax["zt"]; a.clear(); a.grid(True, alpha=0.3)
    a.axhline(sp[2], color="k", lw=1.0, ls=":", label="setpoint")
    for name in CONTROLLERS:
        tr2 = rec.traces.get(name)
        if not tr2 or not len(tr2):
            continue
        d = tr2.arrays()
        a.plot(d["t"], d["z"], STYLES.get(name, "-"), color=COLORS[name], lw=1.3,
               label=LABELS[name])
    a.set_xlabel("t [s]"); a.set_ylabel("$z$ [m]")
    a.set_title("altitude against time", fontsize=9)
    a.margins(y=0.32)
    a.legend(fontsize=6.5, ncol=4, loc="upper center")

    m = rec.metrics()
    bits = "   ".join(
        f"{n}: $\\bar e_z$={v['z_bias']:+.3f} $\\bar e_x$={v['x_bias']:+.3f} "
        f"$\\|e\\|_\\mathrm{{rms}}$={v['p_rms']:.3f} m"
        for n, v in m.items() if n in CONTROLLERS)
    fig.suptitle((title or "S7  hover hold, asymmetric payload") + "\n" + (bits or ""),
                 fontsize=9.5)
    return fig


def render_to(rec, out, title=None):
    """Headless render to a PNG.  Used by --replay and by the periodic save."""
    fig, ax = make_figure(agg=True)
    draw(fig, ax, rec, title=title)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    return out


# --------------------------------------------------------------------------- #
#  the live node
# --------------------------------------------------------------------------- #
def _replay(session, out=None, setpoint=(0.0, 0.0, 1.5), payload=None):
    """Render a finished session's traces without a ROS graph."""
    path = session if session.endswith(".npz") else os.path.join(session,
                                                                 "zx_traces.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no traces at {path}")
    truth = EST.payload_truth(*payload) if payload else None
    rec = DemoRecorder(setpoint=setpoint, truth=truth).load(path)
    out = out or os.path.join(os.path.dirname(path) or ".", "R-F13_zx_demo.png")
    render_to(rec, out, title="S7  hover hold, asymmetric payload")
    print(f"wrote {out}")
    for name, m in rec.metrics().items():
        print(f"  {name:16s} n={m['n']:5d}  e_z={m['z_bias']:+.4f} m  "
              f"e_x={m['x_bias']:+.4f} m  |e|_rms={m['p_rms']:.4f} m  "
              f"|e|_max={m['p_max']:.4f} m")
    return 0


def main(args=None):                                       # pragma: no cover
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="live Z-X plane for the S7 demo")
    ap.add_argument("--replay", default=None,
                    help="a session directory or zx_traces.npz; renders and exits")
    ap.add_argument("--out", default=None)
    ap.add_argument("--payload", nargs=4, type=float, default=None,
                    metavar=("M", "RX", "RY", "RZ"))
    ap.add_argument("--setpoint", nargs=3, type=float, default=[0.0, 0.0, 1.5])
    known, ros_args = ap.parse_known_args(
        [a for a in (args if args is not None else sys.argv[1:])
         if a != "--ros-args"])
    if known.replay:
        return _replay(known.replay, known.out, known.setpoint,
                       known.payload and (known.payload[0], known.payload[1:]))

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import String
    except Exception as ex:                               # noqa: BLE001
        print(f"visualization: no ROS 2 environment ({type(ex).__name__}).  Use "
              f"--replay <session> to render saved traces instead.",
              file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from geometry_msgs.msg import WrenchStamped
    from std_msgs.msg import String
    from acmpc_controller import frames as F
    from acmpc_controller import px4_topics as PT

    class LiveZX(Node):
        def __init__(self):
            super().__init__("live_zx")
            for k, v in (("session_dir", "runs/demo_s7"),
                         ("controller", ""),
                         ("px4_namespace", ""),
                         ("payload_mass", 0.30),
                         ("payload_offset", [0.12, 0.06, -0.04]),
                         ("setpoint", [0.0, 0.0, 1.5]),
                         ("redraw_hz", 4.0),
                         ("save_every_s", 5.0),
                         ("headless", False),
                         ("topic_timeout_s", 20.0)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            self.session = os.path.abspath(g("session_dir"))
            os.makedirs(self.session, exist_ok=True)
            self.npz = os.path.join(self.session, "zx_traces.npz")
            off = PT.as_floats(g("payload_offset"), 3, "payload_offset")
            sp = PT.as_floats(g("setpoint"), 3, "setpoint")
            truth = EST.payload_truth(float(g("payload_mass")), off)
            self.rec = DemoRecorder(setpoint=sp, truth=truth)
            self.rec.load(self.npz)
            if g("controller"):
                self.rec.select(g("controller"))

            # -- PX4 topics, version-resolved at run time (never hard-coded) -- #
            odom = PT.resolve(self, "/fmu/out/vehicle_odometry",
                              timeout_s=float(g("topic_timeout_s")),
                              namespace=g("px4_namespace"))
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
            self.create_subscription(odom.msg_class, odom.topic, self.on_odom, qos)
            self.get_logger().info(f"live_zx: pose from {odom}")
            am = PT.resolve(self, "/fmu/out/actuator_motors", timeout_s=2.0,
                            namespace=g("px4_namespace"), required=False)
            self.thrust = None
            if am is not None:
                self.create_subscription(am.msg_class, am.topic, self.on_motors, qos)
                self.get_logger().info(f"live_zx: thrust from {am}")
            else:
                self.get_logger().warn(
                    "live_zx: actuator_motors absent -- the mass estimate falls "
                    "back to the hover form (9.10), which is what the plot will "
                    "say.  The moment estimate is unaffected.")

            self.create_subscription(WrenchStamped, "/rdp/disturbance_estimate",
                                     self.on_d, 10)
            latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            # transient-local so a plotter started after the driver still learns
            # which controller is flying instead of dropping every sample
            self.create_subscription(String, "/experiment/controller",
                                     self.on_ctrl, latched)

            self.headless = bool(g("headless"))
            self.fig, self.ax = (None, None)
            if not self.headless:
                import matplotlib
                try:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    self.fig, self.ax = make_figure()
                    self.fig.canvas.manager.set_window_title("S7 hover / payload")
                    plt.show(block=False)
                except Exception as ex:                   # noqa: BLE001
                    self.get_logger().warn(
                        f"no interactive matplotlib backend ({type(ex).__name__}: "
                        f"{ex}); falling back to headless PNG updates in "
                        f"{self.session}")
                    self.headless = True
                    matplotlib.use("Agg")
            self.t_ref = None
            self.create_timer(1.0 / max(float(g("redraw_hz")), 0.5), self.redraw)
            self.create_timer(max(float(g("save_every_s")), 1.0), self.save)

        # -- callbacks ------------------------------------------------------ #
        def on_ctrl(self, m):
            name = m.data.strip()
            if name and name != self.rec.active:
                self.save()                    # freeze the finished controller
                self.rec.select(name)
                self.t_ref = None
                self.get_logger().info(f"live_zx: now recording {name!r}")

        def on_odom(self, m):
            if self.rec.active is None:
                return
            t = self.get_clock().now().nanoseconds * 1e-9
            p = F.ned_to_enu_vec(np.asarray(m.position, float))
            q = F.px4_quat_to_enu_flu(np.asarray(m.q, float))
            self.rec.push_pose(t, p, q)
            if self.thrust is not None:
                self.rec.set_estimate(self._last_d(), self.thrust)

        def on_motors(self, m):
            self.thrust = thrust_from_actuator_motors(np.asarray(m.control, float)[:4])

        def _last_d(self):
            tr = self.rec.traces.get(self.rec.active)
            if tr is None or not len(tr):
                return np.zeros(6)
            return np.array([tr.Fx[-1], tr.Fy[-1], tr.Fz[-1],
                             tr.Mx[-1], tr.My[-1], tr.Mz[-1]])

        def on_d(self, m):
            d = np.array([m.wrench.force.x, m.wrench.force.y, m.wrench.force.z,
                          m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z])
            self.rec.set_estimate(d, self.thrust)

        # -- output --------------------------------------------------------- #
        def redraw(self):
            if self.headless:
                return
            try:
                draw(self.fig, self.ax, self.rec)
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception as ex:                       # noqa: BLE001
                self.get_logger().warn(f"redraw failed ({type(ex).__name__}: {ex})")

        def save(self):
            self.rec.save(self.npz)
            if self.headless:
                render_to(self.rec, os.path.join(self.session, "R-F13_zx_demo.png"))

        def finish(self):
            self.save()
            out = render_to(self.rec, os.path.join(self.session,
                                                   "R-F13_zx_demo.png"))
            import json
            with open(os.path.join(self.session, "zx_metrics.json"), "w") as f:
                json.dump(self.rec.metrics(), f, indent=2)
            self.get_logger().info(f"live_zx: wrote {out} and zx_metrics.json")

    rclpy.init(args=ros_args or None)
    node = LiveZX()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
