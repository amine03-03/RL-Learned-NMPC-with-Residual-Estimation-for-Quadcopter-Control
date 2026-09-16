"""xla_probe3.py -- bisect the XLA:CPU crash and test the code-level fix on the
REAL functions.  Run from study/notebooks:   python xla_probe3.py

Where we are: the ~20-line patterns of probe 2 all PASS on the affected box, and
taskset at 1/2/4/8 cores all FAIL, so neither the concatenate shape alone nor
the core count is the trigger.  The trigger needs more of the real program.

So this walks a ladder from fc() up to the full rollout and reports the FIRST
rung that fails -- that rung is the smallest thing that reproduces.  Then it
monkeypatches the concatenates out of the real fc/step_c and re-runs the
failing rung, which answers the only question that matters before touching
verified code: does the rewrite actually fix it?
"""
import sys
import traceback

sys.path.insert(0, "..")
sys.path.insert(0, ".")
import _nbinit            # noqa: F401,E402   enables float64
import jax                # noqa: E402
import jax.numpy as jnp   # noqa: E402
import numpy as np        # noqa: E402
import x500_core_jax as X  # noqa: E402
import study_prelude as S  # noqa: E402


def attempt(label, fn):
    try:
        fn()
        print(f"[PASS] {label}")
        return True
    except Exception as e:
        first = str(e).strip().splitlines()[0][:130] if str(e).strip() else repr(e)
        print(f"[FAIL] {label}\n       {first}")
        return False


# --------------------------------------------------------------------------- #
# concatenate-free replacements, numerically identical by construction
# --------------------------------------------------------------------------- #
def fc_nocat(x, u, d=None, par=None):
    m, T_max = X._ctrl_consts(par)
    v, q = x[..., 3:6], x[..., 6:10]
    om = X._clamp(jnp.asarray(X.OM_MAX) * u[..., 1:4], -X.P.rate_max, X.P.rate_max)
    a_res = jnp.zeros_like(v) if d is None else d[..., 0:3]
    om_res = jnp.zeros_like(om) if d is None else d[..., 3:6]
    acc = (X.thrust_of(u[..., 0:1], T_max) / m * X.qzaxis(q)
           - X.P.g * jnp.asarray(X.E3) + a_res)
    # was: jnp.concatenate([zeros(...,1), om + om_res], -1)
    omq = jnp.zeros(om.shape[:-1] + (4,), om.dtype).at[..., 1:4].set(om + om_res)
    qdot = 0.5 * X.qmul(q, omq)
    # was: jnp.concatenate([v, acc, qdot], -1)
    out = jnp.zeros_like(x)
    out = out.at[..., 0:3].set(v)
    out = out.at[..., 3:6].set(acc)
    out = out.at[..., 6:10].set(qdot)
    return out


def step_c_nocat(x, u, d=None, par=None, dt=None):
    dt = X.P.dt_c if dt is None else dt
    k1 = fc_nocat(x, u, d, par)
    k2 = fc_nocat(x + 0.5 * dt * k1, u, d, par)
    k3 = fc_nocat(x + 0.5 * dt * k2, u, d, par)
    k4 = fc_nocat(x + dt * k3, u, d, par)
    xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # was: jnp.concatenate([xn[..., :6], qnorm(xn[..., 6:10])], -1)
    return xn.at[..., 6:10].set(X.qnorm(xn[..., 6:10]))


def check_equivalence():
    """The rewrite must be a lowering change only, not a maths change."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(8, 10)))
    x = x.at[..., 6:10].set(X.qnorm(x[..., 6:10]))
    u = jnp.asarray(rng.uniform(0.1, 0.9, size=(8, 4)))
    d = jnp.asarray(rng.normal(size=(8, 6)) * 0.1)
    e1 = float(jnp.abs(X.fc(x, u, d) - fc_nocat(x, u, d)).max())
    e2 = float(jnp.abs(X.step_c(x, u, d) - step_c_nocat(x, u, d)).max())
    print(f"  fc     max|orig - rewritten| = {e1:.3e}")
    print(f"  step_c max|orig - rewritten| = {e2:.3e}")
    assert e1 < 1e-12 and e2 < 1e-12, "REWRITE CHANGED THE MATHS -- do not use"
    print("  equivalence OK")


def main():
    print("=== build info ===")
    try:
        jax.print_environment_info()
    except Exception:
        print(f"  jax {jax.__version__}")
    print(f"  jaxlib file: {__import__('jaxlib').__file__}")

    print("\n=== equivalence of the proposed rewrite ===")
    check_equivalence()

    rng = np.random.default_rng(0)
    x8 = jnp.asarray(rng.normal(size=(8, 10)))
    x8 = x8.at[..., 6:10].set(X.qnorm(x8[..., 6:10]))
    u8 = jnp.asarray(rng.uniform(0.1, 0.9, size=(8, 4)))

    print("\n=== ladder: first FAIL is the smallest reproducer ===")
    attempt("1. fc, jitted, (8,10)",
            lambda: jax.block_until_ready(jax.jit(X.fc)(x8, u8)))
    attempt("2. step_c, jitted, (8,10)",
            lambda: jax.block_until_ready(X.step_c(x8, u8)))

    def rung3():
        e0 = jnp.zeros((8, 9))
        du = jnp.zeros((8, 1, 4))
        xr = jnp.zeros((8, 2, 10))
        jax.block_until_ready(X.lin_traj(e0, du, xr)[0])
    attempt("3. lin_traj, N=1", rung3)

    env = S.ev_env("circle", spec=S.nominal_spec(speed=(1.0, 1.5)), seed=202)
    o, e, xr = env.obs()
    _, uref, _ = env.ref_now()

    for ni in (1, 2, 5):
        def rung(ni=ni):
            c = X.make_nmpc_ctrl(N=1, n_iter=ni)
            out = c(o, e, xr, uref=uref)
            jax.block_until_ready(out[0] if isinstance(out, tuple) else out)
        attempt(f"4. single _solve call, N=1, n_iter={ni}", rung)

    # nb1 section [4] sweeps N = (1, 2, 3, 5, 10, 20); a longer horizon builds a
    # much larger HLO module, and the reported failure carried a high
    # instruction id (%slice.5696.7), so horizon is a candidate axis.
    for N in (2, 5, 10, 20):
        def rungN(N=N):
            ev = S.ev_env("circle", spec=S.nominal_spec(speed=(1.0, 1.5)), seed=202)
            c = X.make_nmpc_ctrl(N=N, n_iter=S.CFG["ilqr"])
            r = X.rollout_eval(ev, c, 10, warmup=2)
            jax.block_until_ready(X.stats(r)["rmse"])
        attempt(f"4b. short rollout at N={N} (nb1 [4] sweeps these)", rungN)

    def rung5():
        c = X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"])
        r = X.rollout_eval(env, c, 20, warmup=5)
        jax.block_until_ready(X.stats(r)["rmse"])
    ok_real = attempt("5. real rollout_eval (the known failure)", rung5)

    print("\n=== batch size (does the 8 in f64[8,3] matter?) ===")
    for B in (1, 2, 4, 8, 16):
        def rungB(B=B):
            ev = S.ev_env("circle", spec=S.nominal_spec(speed=(1.0, 1.5)),
                          seed=202, n=B) if _accepts_n() else None
            if ev is None:
                raise RuntimeError("ev_env has no batch argument; skipped")
            c = X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"])
            oo, ee, xx = ev.obs()
            _, ur, _ = ev.ref_now()
            out = c(oo, ee, xx, uref=ur)
            jax.block_until_ready(out[0] if isinstance(out, tuple) else out)
        attempt(f"   batch {B}", rungB)

    print("\n=== THE DECISIVE TEST: same rollout, concatenates removed ===")
    orig_fc, orig_step = X.fc, X.step_c
    try:
        X.fc = fc_nocat
        X.step_c = jax.jit(step_c_nocat, static_argnames=("dt",))
        ok_fix = attempt("6. real rollout_eval with .at[].set() fc/step_c", rung5)
    finally:
        X.fc, X.step_c = orig_fc, orig_step

    print("\n=== verdict ===")
    if not ok_real and ok_fix:
        print("  The code-level rewrite FIXES it. Send me this output and I will")
        print("  apply it to fc/step_c in x500_core_jax.py for real.")
    elif not ok_real and not ok_fix:
        print("  The rewrite does NOT fix it -- the trigger is a different")
        print("  concatenate. Send me this output plus the first FAIL rung above.")
    elif ok_real:
        print("  Nothing failed this run. If the notebook still dies, the trigger")
        print("  is further in (section [4] sweeps N up to 20); send this output.")
    return 0


def _accepts_n():
    import inspect
    try:
        return "n" in inspect.signature(S.ev_env).parameters
    except Exception:
        return False


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
