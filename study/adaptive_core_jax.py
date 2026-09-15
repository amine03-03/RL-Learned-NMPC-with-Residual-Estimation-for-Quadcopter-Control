"""adaptive_core_jax.py -- disturbance scenarios, the residual dynamics
predictor (RDP), and the adaptive environment.

Implements §6 of AGENTS_SPEC_Adaptive_ACMPC.md.  Depends only on
:mod:`x500_core_jax`; no ROS, no PyTorch.

The slung-load tension sign of (6.1) is corrected here (C-3): as written, a
statically hanging load pushes the airframe *up*.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import x500_core_jax as X

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
# §6.1  disturbance scenarios
# --------------------------------------------------------------------------- #
SCEN_LEVELS = {
    "central": (0.0, 0.075, 0.15, 0.25),      # payload mass / vehicle mass
    "asym": (0.0, 0.04, 0.07, 0.11),
    "slung": (0.0, 15.0, 10.0, 5.0, 3.0),     # level = figure-8 period [s]
}
SCEN_RP = {"central": (0.0, 0.0, -0.05), "asym": (0.174, 0.174, -0.05)}
SLUNG_MFRAC, SLUNG_L = 0.149, 0.5
SLUNG_PERIOD = float(2 * np.pi * np.sqrt(SLUNG_L / X.P.g))     # 1.4187 s
SCENARIOS = ("central", "asym", "slung")


def moderate_levels(levels, frac=0.40):
    """§8.5: evaluation levels contracted to ``frac`` of the study brackets.

    The raw brackets put a quarter of the vehicle mass off-centre, which on a
    1.689 thrust-to-weight airframe sits against the actuator limit.  Level 0
    is a mass fraction of zero and is preserved exactly -- **level 0 must be
    flown**: without it you cannot tell conservatism from disturbance
    rejection, and *always read the zero row first*.
    """
    return tuple(round(frac * l, 6) for l in levels)


@dataclass(frozen=True)
class PayloadScenario:
    """A rigidly attached payload: pure vertical force (``central``) or force
    **and** a standing moment (``asym``)."""

    name: str
    level: float = 0.0

    @property
    def r_p(self):
        return np.asarray(SCEN_RP[self.name])

    def sample(self, key, n, level):
        return {}

    def par_kwargs(self, ep, n):
        return dict(payload_m=jnp.full((n,), self.level * X.M_TOT),
                    payload_r=jnp.broadcast_to(jnp.asarray(self.r_p), (n, 3)))

    def analytic_wrench(self):
        """(9.7) the free-parameter-free steady-state prediction [N, N m]."""
        Fz = -self.level * X.M_TOT * X.P.g
        tau = np.cross(self.r_p, np.array([0.0, 0.0, Fz]))
        return np.array([0.0, 0.0, Fz]), tau


@dataclass(frozen=True)
class SlungScenario:
    """A spherical pendulum, **not** a stiff spring.

    A spring cable needs a step well below its natural period and is the usual
    reason such simulations diverge.  With ``n_hat`` the world-frame unit vector
    from the attachment point **to the load**:

        F_ten = m_l ( -g (n_hat . e3) + L ||n_hat_dot||^2 )        (6.1, C-3)
        F_world = F_ten * n_hat

    C-3: the specification writes ``+g (n_hat . e3)``.  Projecting the load's
    equation of motion on ``n_hat`` gives the minus sign, and a hanging load
    (``n_hat = -e3``) must give ``F_ten = +m_l g`` and pull the airframe *down*;
    the specification's sign gives ``-m_l g`` and pushes it up.  The airframe's
    own acceleration coupling ``-a_q . n_hat`` is dropped -- it forms an
    algebraic loop with the airframe acceleration -- which is the specification's
    intent and is declared here rather than hidden.

    The tether produces **no moment only if the attachment point is the CG**;
    for an offset attachment ``tau = r_att x F_body`` is added.
    """

    name: str = "slung"
    level: float = 0.0            # figure-8 excitation period [s]; 0 = detached
    r_att: tuple = (0.0, 0.0, -0.05)
    m_frac: float = SLUNG_MFRAC
    L: float = SLUNG_L

    def sample(self, key, n, level):
        return {}

    def par_kwargs(self, ep, n):
        return {}                 # the load acts through wrench_body, not mass

    @property
    def m_load(self):
        return self.m_frac * X.M_TOT * (self.level > 0)

    def init(self, key, n):
        """Pendulum state: n_hat hanging, with a small random initial swing."""
        nh = jnp.broadcast_to(jnp.asarray([0.0, 0.0, -1.0]), (n, 3))
        v = 0.05 * jax.random.normal(key, (n, 3)) * jnp.asarray([1.0, 1.0, 0.0])
        return jnp.concatenate([nh, v - nh * jnp.sum(v * nh, -1, keepdims=True)], -1)

    def wrench_body(self, pend, s, t):
        """Body-frame [F(3), tau(3)] applied to the airframe."""
        n_hat, n_dot = pend[:, 0:3], pend[:, 3:6]
        ml = self.m_load
        F_ten = ml * (-X.P.g * n_hat[:, 2:3] + self.L * jnp.sum(n_dot ** 2, -1, keepdims=True))
        F_w = F_ten * n_hat
        R = X.qrotmat(s[:, X.SQ])
        F_b = jnp.einsum("bji,bj->bi", R, F_w)          # R^T F_world
        tau_b = jnp.cross(jnp.broadcast_to(jnp.asarray(self.r_att), F_b.shape), F_b)
        return jnp.concatenate([F_b, tau_b], -1)

    def advance(self, pend, n_sub=None, dt=None):
        """Semi-implicit (velocity-first) integration on the unit sphere.

        Explicit Euler on an oscillator injects energy monotonically; the drift
        is logged per episode and asserted below 2 % (T-19).
        """
        dt = X.P.dt_c if dt is None else dt
        n_sub = X.P.n_sub if n_sub is None else n_sub
        h = dt / n_sub

        def body(_, st):
            nh, nd = st[:, 0:3], st[:, 3:6]
            g_perp = -X.P.g * (jnp.asarray(X.E3) - nh * nh[:, 2:3]) / self.L
            acc = g_perp - jnp.sum(nd ** 2, -1, keepdims=True) * nh
            nd = nd + h * acc                                    # velocity first
            nh = nh + h * nd                                     # then position
            nh = nh / jnp.linalg.norm(nh, axis=-1, keepdims=True)
            nd = nd - nh * jnp.sum(nd * nh, -1, keepdims=True)    # keep n_dot _|_ n
            return jnp.concatenate([nh, nd], -1)

        return jax.lax.fori_loop(0, n_sub, body, pend)

    def energy(self, pend):
        """Tether energy per unit load mass: 1/2 L^2 ||n_dot||^2 + g L (n.e3)."""
        nh, nd = pend[:, 0:3], pend[:, 3:6]
        return 0.5 * self.L ** 2 * jnp.sum(nd ** 2, -1) + X.P.g * self.L * nh[:, 2]

    def excite(self, pend, t, n):
        """Figure-8 excitation at the scenario period: shorter = harsher."""
        if self.level <= 0:
            return pend
        w = 2 * np.pi / self.level
        amp = 0.35
        drive = amp * w * jnp.stack(
            [jnp.cos(w * t), 2 * jnp.cos(2 * w * t), jnp.zeros_like(t)], -1)
        nh, nd = pend[:, 0:3], pend[:, 3:6]
        nd = nd + X.P.dt_c * drive / self.L
        nd = nd - nh * jnp.sum(nd * nh, -1, keepdims=True)
        return jnp.concatenate([nh, nd], -1)


def make_scenario(name, level):
    if name in ("central", "asym"):
        return PayloadScenario(name, float(level))
    if name == "slung":
        return SlungScenario(level=float(level))
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# §6.2  the residual dynamics predictor
# --------------------------------------------------------------------------- #
FRAME_DIM = 26          # eq (6.2)
H_DEFAULT = 32          # 0.64 s at 50 Hz -- matches the ROS deployment (§9.5)
H_BENCH = (16, 32, 64, 128)
OUT_DIM = 6
CHANNEL_NAMES = ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z")
CHANNEL_UNITS = ("N", "N", "N", "N.m", "N.m", "N.m")

# The PWM block of (6.2) is the key design decision of the whole adaptive half.
# With an integrating inner rate loop a *constant* external moment is driven out
# of the rate error in steady state, while the mixer output holds a spread.
# Formally, as om -> 0 and om_dot -> 0, (4.11) reduces to
#     tau_ext -> -Mtau_nom K_T Omega^2,
# built entirely from the motor commands.  State and setpoint alone cannot
# resolve a standing moment; the actuator commands can.  If the motor-command
# topic is not published the moment channels are unobservable and the dataset is
# worthless -- §9.4 check 4 refuses to start a moment-producing scenario then.


def _glorot(key, shape, gain=1.0):
    fan_in, fan_out = shape[0], shape[-1]
    lim = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def _dense_init(key, nin, nout, gain=1.0):
    return {"W": _glorot(key, (nin, nout), gain), "b": jnp.zeros((nout,))}


def _dense(p, x):
    return x @ p["W"] + p["b"]


# -- GRU (Cho et al. 2014) --------------------------------------------------- #
def gru_cell_init(key, nin, nh):
    ks = jax.random.split(key, 2)
    return {"Wi": _glorot(ks[0], (nin, 3 * nh)), "Wh": _glorot(ks[1], (nh, 3 * nh)),
            "bi": jnp.zeros((3 * nh,)), "bh": jnp.zeros((3 * nh,)), "nh": nh}


def gru_scan(p, xs):
    """xs (T,B,nin) -> hs (T,B,nh).  Gate equations written out explicitly, so
    the weights transplant into the pure-NumPy ``rdp_infer`` and T-17 can check
    parity rather than trust a library layer."""
    nh = p["nh"]

    def step(h, x):
        gi = x @ p["Wi"] + p["bi"]
        gh = h @ p["Wh"] + p["bh"]
        ir, iz, inn = gi[:, :nh], gi[:, nh:2 * nh], gi[:, 2 * nh:]
        hr, hz, hn = gh[:, :nh], gh[:, nh:2 * nh], gh[:, 2 * nh:]
        r = jax.nn.sigmoid(ir + hr)
        z = jax.nn.sigmoid(iz + hz)
        n = jnp.tanh(inn + r * hn)
        h = (1.0 - z) * n + z * h
        return h, h

    h0 = jnp.zeros((xs.shape[1], nh))
    _, hs = jax.lax.scan(step, h0, xs)
    return hs


# -- LSTM (Hochreiter & Schmidhuber 1997) ------------------------------------ #
def lstm_cell_init(key, nin, nh):
    ks = jax.random.split(key, 2)
    b = jnp.zeros((4 * nh,)).at[nh:2 * nh].set(1.0)      # forget-gate bias = 1
    return {"Wi": _glorot(ks[0], (nin, 4 * nh)), "Wh": _glorot(ks[1], (nh, 4 * nh)),
            "bi": b, "bh": jnp.zeros((4 * nh,)), "nh": nh}


def lstm_scan(p, xs):
    nh = p["nh"]

    def step(carry, x):
        h, c = carry
        g = x @ p["Wi"] + p["bi"] + h @ p["Wh"] + p["bh"]
        i = jax.nn.sigmoid(g[:, :nh])
        f = jax.nn.sigmoid(g[:, nh:2 * nh])
        gg = jnp.tanh(g[:, 2 * nh:3 * nh])
        o = jax.nn.sigmoid(g[:, 3 * nh:])
        c = f * c + i * gg
        h = o * jnp.tanh(c)
        return (h, c), h

    z = jnp.zeros((xs.shape[1], nh))
    _, hs = jax.lax.scan(step, (z, z), xs)
    return hs


# -- causal convolutions ----------------------------------------------------- #
def _causal_conv(p, x, dilation):
    """x (B,T,C) -> (B,T,C_out), left-padded so output t sees only inputs <= t."""
    k = p["W"].shape[0]
    pad = (k - 1) * dilation
    xp = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
    y = jax.lax.conv_general_dilated(
        xp, p["W"], window_strides=(1,), padding="VALID", rhs_dilation=(dilation,),
        dimension_numbers=("NWC", "WIO", "NWC"))
    return y + p["b"]


def tcn_receptive_field(n_block, k):
    """1 + sum over blocks of 2*(k-1)*dilation -- two convolutions per block."""
    return 1 + sum(2 * (k - 1) * 2 ** i for i in range(n_block))


def tcn_blocks_for(H, k=3):
    """Smallest block count whose receptive field covers the whole window.

    With 3 blocks and k=3 the field is 29 frames; at H=32 the first three frames
    of the window are then invisible to the model, which looks like a dead
    encoder rather than a design choice.  Sized from H instead.
    """
    n = 1
    while tcn_receptive_field(n, k) < H and n < 12:
        n += 1
    return n


def tcn_init(key, nin, nh, n_block=3, k=3):
    ks = jax.random.split(key, 2 * n_block)
    blocks = []
    c = nin
    for i in range(n_block):
        blocks.append({
            "c1": {"W": _glorot(ks[2 * i], (k, c, nh)), "b": jnp.zeros((nh,))},
            "c2": {"W": _glorot(ks[2 * i + 1], (k, nh, nh)), "b": jnp.zeros((nh,))},
            "res": None if c == nh else {"W": _glorot(ks[2 * i], (1, c, nh)),
                                         "b": jnp.zeros((nh,))},
            "dil": 2 ** i})
        c = nh
    return {"blocks": blocks, "nh": nh}


def tcn_apply(p, x):
    """Dilated causal residual stack (Bai, Kolter & Koltun 2018)."""
    for blk in p["blocks"]:
        h = jax.nn.relu(_causal_conv(blk["c1"], x, blk["dil"]))
        h = jax.nn.relu(_causal_conv(blk["c2"], h, blk["dil"]))
        res = x if blk["res"] is None else _causal_conv(blk["res"], x, 1)
        x = h + res
    return x


def cnn_receptive_field(n_layer, k):
    """1 + (k-1)*(2^L - 1): each layer sees k taps at stride 2^i."""
    return 1 + (k - 1) * (2 ** n_layer - 1)


def cnn_layers_for(H, k=5):
    """Smallest layer count covering the window, for the same reason as the TCN."""
    n = 1
    while cnn_receptive_field(n, k) < H and n < 10:
        n += 1
    return n


def cnn_init(key, nin, nh, n_layer=3, k=5):
    ks = jax.random.split(key, n_layer)
    layers, c = [], nin
    for i in range(n_layer):
        layers.append({"W": _glorot(ks[i], (k, c, nh)), "b": jnp.zeros((nh,))})
        c = nh
    return {"layers": layers, "nh": nh}


def cnn_apply(p, x):
    """Strided causal convolutions.

    The decimation is anchored to the **right** edge: ``x[:, ::2]`` keeps
    indices 0, 2, 4, ... and therefore *drops the most recent frame* whenever
    the length is even, which for H = 32 makes the predictor blind to the very
    sample it is supposed to react to.  Starting at ``(T-1) % 2`` keeps index
    T-1 in every case.
    """
    for i, lyr in enumerate(p["layers"]):
        x = jax.nn.relu(_causal_conv(lyr, x, 1))
        if i < len(p["layers"]) - 1 and x.shape[1] >= 4:
            x = x[:, (x.shape[1] - 1) % 2::2]
    return x


# -- the predictor ----------------------------------------------------------- #
ENCODERS = ("GRU", "LSTM", "TCN", "CNN")


def rdp_init(key, kind="GRU", H=H_DEFAULT, nin=FRAME_DIM, hid=(128, 64), out=OUT_DIM,
             **kw):
    """Default deployment architecture (§9.5): normalise -> GRU(128) -> GRU(64)
    -> Linear(6)."""
    ks = jax.random.split(key, 6)
    p = {"kind": kind, "H": H, "nin": nin, "out": out, "hid": tuple(hid)}
    if kind == "GRU":
        p["l1"] = gru_cell_init(ks[0], nin, hid[0])
        p["l2"] = gru_cell_init(ks[1], hid[0], hid[1])
        p["head"] = _dense_init(ks[2], hid[1], out, gain=0.1)
    elif kind == "LSTM":
        p["l1"] = lstm_cell_init(ks[0], nin, hid[0])
        p["l2"] = lstm_cell_init(ks[1], hid[0], hid[1])
        p["head"] = _dense_init(ks[2], hid[1], out, gain=0.1)
    elif kind == "TCN":
        kk = kw.get("k", 3)
        p["enc"] = tcn_init(ks[0], nin, hid[1],
                            n_block=kw.get("n_block", tcn_blocks_for(H, kk)), k=kk)
        p["rf"] = tcn_receptive_field(len(p["enc"]["blocks"]), kk)
        p["head"] = _dense_init(ks[2], hid[1], out, gain=0.1)
    elif kind == "CNN":
        kk = kw.get("k", 5)
        p["enc"] = cnn_init(ks[0], nin, hid[1],
                            n_layer=kw.get("n_layer", cnn_layers_for(H, kk)), k=kk)
        p["rf"] = cnn_receptive_field(len(p["enc"]["layers"]), kk)
        p["head"] = _dense_init(ks[2], hid[1], out, gain=0.1)
    else:
        raise ValueError(kind)
    return p


def rdp_apply(p, win, mu, sd):
    """win (B,H,26) -> (B,6) **normalised** output.

    The caller multiplies by the output scale to get physical units; the loss
    (6.3) is computed in this normalised space, because the first three targets
    are in N and the last three in N.m and an unnormalised squared error fits
    only the larger block.
    """
    x = (win - mu) / sd
    kind = p["kind"]
    if kind in ("GRU", "LSTM"):
        scan = gru_scan if kind == "GRU" else lstm_scan
        xs = jnp.swapaxes(x, 0, 1)                 # (H,B,26)
        h1 = scan(p["l1"], xs)
        h2 = scan(p["l2"], h1)
        feat = h2[-1]                              # causal: the last frame only
    else:
        enc = tcn_apply if kind == "TCN" else cnn_apply
        feat = enc(p["enc"], x)[:, -1]
    return _dense(p["head"], feat)


def rdp_loss(p, win, tgt_n, mu, sd):
    """(6.3) per-channel-normalised MSE."""
    pred = rdp_apply(p, win, mu, sd)
    return jnp.mean((pred - tgt_n) ** 2)


def rdp_param_count(p):
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda z: z if hasattr(z, "size") else None, p))
    return int(sum(getattr(z, "size", 0) for z in leaves))


def r2_score(y, yh):
    """Per-channel coefficient of determination; 0 is break-even against the mean."""
    ss_res = ((y - yh) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-30), np.nan)


def ckpt_entry(name, params, kind, kw, mu, sd, out_sd, H, r2):
    """§6.2 checkpoint format: enough to rebuild without guessing."""
    return {name: {"state": jax.tree_util.tree_map(
        lambda z: np.asarray(z) if hasattr(z, "shape") else z, params),
        "kind": kind, "kw": dict(kw),
        "scales": {"mu": np.asarray(mu), "sd": np.asarray(sd),
                   "out_sd": np.asarray(out_sd)},
        "H": int(H), "r2": [float(v) for v in r2]}}


def rebuild_rdp(entry):
    """Rebuild a predictor from a checkpoint entry -- ctor kwargs are stored."""
    p = jax.tree_util.tree_map(
        lambda z: jnp.asarray(z) if isinstance(z, np.ndarray) else z, entry["state"])
    return p, entry["scales"], entry["H"]


# --------------------------------------------------------------------------- #
# §6.2  training the predictor
# --------------------------------------------------------------------------- #
def make_windows(frames, targets, H, ep_id=None):
    """Sliding windows of length H, **never straddling an episode boundary**.

    frames (T,B,26), targets (T,B,6) -> (n,H,26), (n,6).  The first H-1 samples
    of every episode are dropped (§9.6); with ``ep_id`` supplied (T,B) a window
    is kept only when all H of its frames carry the same episode id, so a
    respawn mid-rollout cannot leak one flight into another.
    """
    T, B = frames.shape[0], frames.shape[1]
    W, Y = [], []
    for k in range(H - 1, T):
        if ep_id is not None:
            ok = np.all(ep_id[k - H + 1:k + 1] == ep_id[k], axis=0)
            if not ok.any():
                continue
            W.append(frames[k - H + 1:k + 1].transpose(1, 0, 2)[ok])
            Y.append(targets[k][ok])
        else:
            W.append(frames[k - H + 1:k + 1].transpose(1, 0, 2))
            Y.append(targets[k])
    if not W:
        return np.zeros((0, H, frames.shape[-1])), np.zeros((0, targets.shape[-1]))
    return np.concatenate(W, 0), np.concatenate(Y, 0)


def split_by_episode(n_ep, seed=0, fracs=(0.70, 0.15, 0.15)):
    """Split **by complete episode, never by shuffled samples** (§9.6)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_ep)
    a = int(fracs[0] * n_ep); b = a + int(fracs[1] * n_ep)
    return idx[:a], idx[a:b], idx[b:]


def train_rdp(key, kind, Wtr, Ytr, Wva, Yva, H=H_DEFAULT, epochs=60, batch=4096,
              lr=1e-3, hid=(128, 64), verbose=True, **kw):
    """Fit one encoder on the identical window and target, so the comparison
    isolates the encoder."""
    import optax
    mu = jnp.asarray(Wtr.reshape(-1, Wtr.shape[-1]).mean(0))
    sd = jnp.asarray(Wtr.reshape(-1, Wtr.shape[-1]).std(0) + 1e-6)
    out_sd = jnp.asarray(Ytr.std(0) + 1e-9)                    # s_j of (6.3)
    p = rdp_init(key, kind, H=H, nin=Wtr.shape[-1], hid=hid, **kw)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    trainable = {k: v for k, v in p.items() if k in ("l1", "l2", "enc", "head")}
    static = {k: v for k, v in p.items() if k not in trainable}
    st = opt.init(trainable)
    Wtr_j, Ytr_n = jnp.asarray(Wtr), jnp.asarray(Ytr) / out_sd
    Wva_j, Yva_n = jnp.asarray(Wva), jnp.asarray(Yva) / out_sd

    @jax.jit
    def upd(tr, st, w, y):
        loss, g = jax.value_and_grad(lambda t: rdp_loss({**static, **t}, w, y, mu, sd))(tr)
        u, st = opt.update(g, st, tr)
        return optax.apply_updates(tr, u), st, loss

    @jax.jit
    def val(tr):
        return rdp_loss({**static, **tr}, Wva_j, Yva_n, mu, sd)

    n = Wtr.shape[0]
    rng = np.random.default_rng(0)
    hist = []
    best, best_tr = np.inf, trainable
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            trainable, st, l = upd(trainable, st, Wtr_j[idx], Ytr_n[idx])
            tot += float(l) * len(idx)
        v = float(val(trainable))
        hist.append(dict(epoch=ep, train=tot / n, val=v))
        if v < best:
            best, best_tr = v, jax.tree_util.tree_map(lambda z: z, trainable)
        if verbose and (ep % max(epochs // 6, 1) == 0 or ep == epochs - 1):
            print(f"    {kind} ep {ep:3d}  train {tot/n:.4f}  val {v:.4f}")
    p = {**static, **best_tr}
    return p, {"mu": mu, "sd": sd, "out_sd": out_sd}, hist


def rdp_predict(p, scales, W, batch=8192):
    """Physical-unit predictions for a stack of windows."""
    out = []
    for i in range(0, W.shape[0], batch):
        out.append(np.asarray(rdp_apply(p, jnp.asarray(W[i:i + batch]),
                                        scales["mu"], scales["sd"])
                              * scales["out_sd"]))
    return np.concatenate(out, 0) if out else np.zeros((0, OUT_DIM))


def rdp_latency_ms(p, scales, H, reps=60):
    """**Single-window** inference latency -- admissibility is decided by this
    against the 20 ms period, not by R^2 (§8.5)."""
    import time
    w = jnp.zeros((1, H, FRAME_DIM))
    f = jax.jit(lambda ww: rdp_apply(p, ww, scales["mu"], scales["sd"]))
    jax.block_until_ready(f(w))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(w))
        ts.append(1e3 * (time.perf_counter() - t0))
    return {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95))}


# --------------------------------------------------------------------------- #
# §6.3  AdaptEnv
# --------------------------------------------------------------------------- #
WRENCH_DR_SWEEP = (0.05, 0.10, 0.20, 0.32)      # fractions of vehicle weight
WRENCH_DR_START = 0.10                          # **do not start at 0.32**


def wrench_dr_magnitudes(frac):
    """F_DR = frac*m*g [N];  tau_DR = 0.30625*frac*m*g*arm [N m].

    The ratio is fixed by the §4.4 worked example: at frac = 0.32 the
    specification quotes F = 6.478 N and tau = 0.098*m*g*0.174 = 0.3452 N m, so
    the moment coefficient is 0.098/0.32 = 0.30625 of the force coefficient.
    """
    mg = X.M_TOT * X.P.g
    return float(frac * mg), float(0.30625 * frac * mg * 0.174)


class AdaptEnv(X.Env):
    """:class:`x500_core_jax.Env` plus a disturbance scenario, an optional
    estimator and a rolling causal window buffer.

    Attaching ``env.estimator`` makes :meth:`d_channel` return the **prediction**
    instead of the truth; :meth:`d_truth` always returns the truth and is for
    the logger and the evaluator only.  The window the estimator reads is built
    from a ring buffer the disturbance manager cannot write to (§9.1), so no
    injected disturbance value can leak into it.
    """

    def __init__(self, n, seed, ep_len, dist, scen, level, oracle=False,
                 wrench_dr=False, mixer="nominal", H=H_DEFAULT, paths=("circle",),
                 **kw):
        self.scen_name, self.level_val = scen, level
        sc = make_scenario(scen, level)
        self.wrench_dr = wrench_dr
        self.H = H
        self._buf = None
        super().__init__(n, seed, ep_len, dist, paths, oracle=oracle, mixer=mixer,
                         scen=sc, level=level, **kw)

    # -- scenario state ----------------------------------------------------- #
    def reset(self, key=None):
        out = super().reset(key)
        self.key, k = jax.random.split(self.key)
        if isinstance(self.scen, SlungScenario):
            self.pend = self.scen.init(k, self.n)
            self.E0 = self.scen.energy(self.pend)
        else:
            self.pend = None
        if self.wrench_dr:
            self._sample_wrench_dr()
        self._buf = np.zeros((self.H, self.n, FRAME_DIM))
        self._buf_fill = 0
        self._apply_scenario_wrench()
        return out

    def _sample_wrench_dr(self):
        """Training-time wrench randomisation (§6.3).  Start at 0.10, not 0.32:
        0.32 is 32 % of vehicle weight and saturates training (§5.13 gate)."""
        self.key, k1, k2 = jax.random.split(self.key, 3)
        F, T = wrench_dr_magnitudes(self.wrench_dr if isinstance(self.wrench_dr, float)
                                    else WRENCH_DR_START)
        d = jax.random.normal(k1, (self.n, 3))
        d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
        t = jax.random.normal(k2, (self.n, 3))
        t = t / jnp.linalg.norm(t, axis=-1, keepdims=True)
        self._dr_wrench = jnp.concatenate([F * d, T * t], -1)

    def _apply_scenario_wrench(self):
        w = jnp.zeros((self.n, 6))
        if self.pend is not None:
            w = w + self.scen.wrench_body(self.pend, self.state, self.t)
        if self.wrench_dr:
            w = w + self._dr_wrench
        self.par = {**self.par, "wrench_body": w}

    # -- the causal window -------------------------------------------------- #
    def push_frame(self):
        """Append the current 26-D frame (6.2) to the ring buffer."""
        f = np.asarray(self.frame26())
        self._buf = np.concatenate([self._buf[1:], f[None]], 0)
        self._buf_fill = min(self._buf_fill + 1, self.H)

    def window(self):
        """(B,H,26).  Warm-up frames are the zero-padded buffer; ``ready()``
        says whether H real frames are present."""
        return self._buf.transpose(1, 0, 2)

    def ready(self):
        return self._buf_fill >= self.H

    def attach(self, params, scales, filt=0.0):
        """Attach a predictor: ``d_channel()`` then returns the estimate.

        ``filt`` is the optional first-order causal filter of §9.11 step 4.
        """
        st = {"y": None}
        f = jax.jit(lambda w: rdp_apply(params, w, scales["mu"], scales["sd"])
                    * scales["out_sd"])

        def est(env):
            y = f(jnp.asarray(env.window()))
            y = jnp.where(env.ready(), y, 0.0)
            if filt > 0.0 and st["y"] is not None:
                y = filt * st["y"] + (1 - filt) * y
            st["y"] = y
            return y

        self.estimator = est
        return self

    # -- transition --------------------------------------------------------- #
    def step(self, u):
        self.push_frame()
        self._apply_scenario_wrench()
        r, done, info = super().step(u)
        if self.pend is not None:
            self.pend = self.scen.advance(self.pend)
            self.pend = self.scen.excite(self.pend, self.t, self.n)
            rs = info["reset"]
            if bool(jnp.any(rs)):
                self.key, k = jax.random.split(self.key)
                new = self.scen.init(k, self.n)
                self.pend = jnp.where(rs[:, None], new, self.pend)
                self.E0 = jnp.where(rs, self.scen.energy(new), self.E0)
            info["tether_drift"] = jnp.abs(
                self.scen.energy(self.pend) - self.E0) / jnp.abs(self.E0)
        if self.wrench_dr and bool(jnp.any(info["reset"])):
            self._sample_wrench_dr()
        return r, done, info
