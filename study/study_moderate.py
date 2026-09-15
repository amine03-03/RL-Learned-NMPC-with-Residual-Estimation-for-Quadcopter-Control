"""study_moderate.py -- disturbance moderation, suite comparison and the
on-path start used by the video notebook.  Implements §7.2.

Why moderation exists: the raw evaluation suites reach 15 m/s (in-distribution)
and 25 m/s (out-of-distribution) of wind against a 13.35 m/s^2 lateral budget.
The symptom is unambiguous -- 41-78 % actuator saturation, every controller
within 0.3 m of every other, ``maxerr`` pinned at the 3 m bound, and the
circle/fig8/square columns agreeing to 0.03 m.  **When the path stops mattering,
you are measuring the wind.**
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import x500_core_jax as X

#: keys that are copied through untouched.  **Never contract ``speed``** -- it
#: would change the reference the controllers are scored against, so the
#: moderated suite would no longer be a statement about the same task.
PASSTHROUGH = ("speed", "paths", "kind", "task", "ep_kind", "noise", "seed")
WIND_KEYS = ("wind", "gust", "vw")
PHI_DEFAULT = 0.15


def moderate(spec, frac=PHI_DEFAULT, wind=(0.0, 2.0)):
    """(7.1) contract each multiplicative band toward its nominal 1.

        (l, h) -> (1 + (l-1)*phi,  1 + (h-1)*phi)

    Contracting *preserves the shape* of the suite, so the moderated version is
    a strict subset and results remain statements about the same disturbance
    structure.  The wind band is replaced by an absolute mild one rather than
    contracted, because its nominal is 0 and (7.1) would leave it unchanged.
    """
    out = {}
    for k, v in spec.items():
        if k in PASSTHROUGH:
            out[k] = v
        elif k in WIND_KEYS:
            out[k] = tuple(float(z) for z in wind)
        else:
            lo, hi = v
            out[k] = (1.0 + (lo - 1.0) * frac, 1.0 + (hi - 1.0) * frac)
    return out


def check_moderate(before, after, wind_cap=None):
    """Assert the three invariants of (7.1) and that ``speed`` was not touched.

    The invariants are stated carefully, because the obvious one is wrong.  §7.2
    says the moderated suite is *"a strict subset"* of the raw one -- true only
    for a band that straddles the nominal 1.  Every S3 multiplicative band lies
    entirely **above** 1 (drag 2.20-3.00, lag 3.50-5.00), so contracting toward
    1 moves it *outside* the raw band, toward the nominal.  A containment test
    written the naive way fails on a correct moderation (N-6).

    What (7.1) actually guarantees, for every band:

    1. **side preservation** -- each endpoint stays on its own side of 1, so the
       suite keeps its shape and the moderated result is still a statement about
       the same disturbance structure;
    2. **narrowing** -- the band is no wider than before;
    3. **containment in the hull** -- the moderated band lies inside
       ``hull(raw band, {1})``, i.e. between the nominal and the raw demand.

    Raises naming the offending key.  A silent contraction of ``speed`` would
    change the reference the controllers are scored against and invalidate every
    comparison built on the suite.
    """
    for k in PASSTHROUGH:
        if k in before or k in after:
            if before.get(k) != after.get(k):
                raise AssertionError(
                    f"moderation touched passthrough key {k!r}: "
                    f"{before.get(k)!r} -> {after.get(k)!r}")
    for k, v in before.items():
        if k in PASSTHROUGH:
            continue
        if k not in after:
            raise AssertionError(f"moderated spec is missing key {k!r}")
        lo, hi = v
        alo, ahi = after[k]
        if k in WIND_KEYS:
            if wind_cap is not None and ahi > wind_cap + 1e-12:
                raise AssertionError(f"wind band {k!r} exceeds cap: {ahi} > {wind_cap}")
            if ahi > hi + 1e-12:
                raise AssertionError(f"wind band {k!r} was widened: {ahi} > {hi}")
            continue
        for nm, raw, mod in (("lo", lo, alo), ("hi", hi, ahi)):
            if np.sign(round(raw - 1.0, 12)) != np.sign(round(mod - 1.0, 12)):
                raise AssertionError(
                    f"moderated band {k!r} changed side at the {nm} endpoint: "
                    f"{raw} -> {mod} across the nominal 1")
        if (ahi - alo) > (hi - lo) + 1e-12:
            raise AssertionError(f"moderated band {k!r} is not narrower: "
                                 f"{ahi-alo} > {hi-lo}")
        h_lo, h_hi = min(lo, 1.0), max(hi, 1.0)
        if not (h_lo - 1e-12 <= alo and ahi <= h_hi + 1e-12):
            raise AssertionError(
                f"moderated band {k!r} escapes hull(raw, 1): "
                f"({alo},{ahi}) not inside ({h_lo},{h_hi})")
    return True


def spec_diff(before, after):
    """A before/after table of what moderation changed."""
    rows = []
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        if isinstance(b, tuple) and isinstance(a, tuple) and len(b) == 2:
            rows.append(dict(channel=k, before=f"{b[0]:g}-{b[1]:g}",
                             after=f"{a[0]:g}-{a[1]:g}",
                             width_before=b[1] - b[0], width_after=a[1] - a[0],
                             ratio=(a[1] - a[0]) / (b[1] - b[0]) if b[1] > b[0] else np.nan))
        else:
            rows.append(dict(channel=k, before=str(b), after=str(a),
                             width_before=np.nan, width_after=np.nan, ratio=np.nan))
    return pd.DataFrame(rows)


def disjointness(s2, s3):
    """Which moderated channels are actually disjoint between two suites.

    C-4: the specification claims the moderated suites are *"disjoint on drag,
    lag, thrust and wind but overlap on mass"*.  They are not.  Wherever the raw
    S3 band is a **superset** of the raw S2 band -- true for mass, thrust, K_w
    and inertia -- contraction toward the common centre 1 preserves the
    inclusion.  Only drag, lag and wind stay separated.  This function computes
    the table rather than asserting the claim, so the notebooks print what is
    true of the suites actually in use.
    """
    rows = []
    for k in sorted(set(s2) & set(s3)):
        if k in PASSTHROUGH or not isinstance(s2[k], tuple):
            continue
        a, b = s2[k]; c, d = s3[k]
        rows.append(dict(channel=k, S2=f"{a:.3f}-{b:.3f}", S3=f"{c:.3f}-{d:.3f}",
                         disjoint=bool(b < c or d < a),
                         S2_subset_of_S3=bool(c <= a and b <= d)))
    return pd.DataFrame(rows)


def make_lltc_ctrl(X_mod, model, N=1, n_iter=25):
    """Rebuild the LLTC controller **explicitly** (§8.2 checkpoint note).

    ``lltc/model.pkl`` holds a *terminal cost network, not an actor*.  Any
    generic ``controllers()`` loader that keys on ``'actor'`` raises KeyError and
    silently skips it, which is worse than failing: the comparison then quietly
    has one fewer row.
    """
    import jax
    import jax.numpy as jnp
    state = {"env": None}
    eps = model.get("ltc", {}).get("eps", 1e-3)
    params = model["state"]

    def P_of(e):
        L = lltc_matrix(params, e, eps)
        return L

    @jax.jit
    def _solve(e, xr_seq, uref_seq):
        Pt = P_of(e)
        return X_mod.solve_quad(e, xr_seq, N, Pt, None, None, n_iter,
                                uref_seq=uref_seq)

    def f(o, e, xr, uref=None, d=None):
        ev = state["env"]
        if ev is not None:
            xr_seq, uref_seq = ev.ref_traj(N), ev.ref_useq(N)
        else:
            xr_seq = jnp.broadcast_to(xr[:, None], (e.shape[0], N + 1, X_mod.NX))
            uref_seq = X_mod.uref_from_traj(xr_seq)
        return _solve(e, xr_seq, uref_seq), {}

    f.bind_env = lambda ev: state.__setitem__("env", ev)
    f.name = f"LLTC N={N}"
    return f


def lltc_matrix(params, e, eps=1e-3):
    """(8.1)  P_theta(e) = L(e) L(e)^T + eps I, L lower triangular from an MLP.

    PSD by construction, so (5.2) stays well posed for any theta.
    """
    import jax.numpy as jnp
    z = X.mlp_apply(params, e)
    n = X.NE
    tri = np.tril_indices(n)
    L = jnp.zeros((e.shape[0], n, n)).at[:, tri[0], tri[1]].set(z)
    return jnp.einsum("bij,bkj->bik", L, L) + eps * jnp.eye(n)


def lltc_init(key, hid=128, n_layer=2, eps=1e-3):
    n_out = X.NE * (X.NE + 1) // 2         # 45 free entries
    return X.mlp_init(key, [X.NE] + [hid] * n_layer + [n_out], scale_last=0.05)


def start_error(env):
    """Current ||p - p_ref|| per vehicle."""
    import jax.numpy as jnp
    xr, _, _ = env.ref_now()
    return jnp.linalg.norm(X.plant_to_ctrl(env.state)[:, 0:3] - xr[:, 0:3], axis=-1)


def settle_on_path(env, pilot, steps=150, tol=0.05, strict=False, verbose=True,
                   label=""):
    """Fly a **fixed reference pilot** until the vehicle is on the path.

    The environment seeds a ~0.5 m offset, so an unmodified clip opens with a
    recovery transient rather than tracking.

    > The pilot must be a FIXED reference controller, **not** the controller
    > under test.  Using the controller under test is circular: one that cannot
    > hold the path drives the error *up* during the warm-up, so it can never be
    > made to start on the path, and its clip then shows divergence from
    > wherever it drifted to.  A fixed pilot also gives every clip in a
    > comparison an identical initial state.

    Returns ``(e_inject, e_start)``.
    """
    import jax.numpy as jnp
    e_inject = float(jnp.max(start_error(env)))
    if hasattr(pilot, "bind_env"):
        pilot.bind_env(env)
    if hasattr(pilot, "reset"):
        pilot.reset()
    for k in range(steps):
        o, e, xr = env.obs()
        _, uref, _ = env.ref_now()
        out = pilot(o, e, xr, uref=uref)
        env.step(out[0] if isinstance(out, tuple) else out)
        if float(jnp.max(start_error(env))) < tol:
            break
    e_start = float(jnp.max(start_error(env)))
    if verbose:
        print(f"  settle{' ' + label if label else ''}: "
              f"e_inject={e_inject:.3f} m -> e_start={e_start:.4f} m "
              f"in {k+1} steps (tol {tol})")
    if e_start > tol:
        msg = (f"pilot did not reach the path: e_start={e_start:.3f} > tol={tol}. "
               "The pilot is a fixed reference controller, so this is a problem "
               "with the reference or the plant, not with the controller under test.")
        if strict:
            raise RuntimeError(msg)
        print("  !! " + msg)
    return e_inject, e_start
