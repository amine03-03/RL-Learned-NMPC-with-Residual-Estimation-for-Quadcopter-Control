"""xla_probe2.py -- minimal reproducer + candidate fixes for the XLA:CPU
concatenate tiling crash.  Run from study/notebooks:  python xla_probe2.py

probe 1 found no XLA_FLAGS workaround, and every flag failed with a BYTE
IDENTICAL error (same HLO id), meaning the failure is in an HLO pass that runs
before those flags apply.  This script attacks it from the other side: it
reproduces the pattern in ~20 lines, then tests fixes we can apply in the
repository itself, which would work on every machine rather than needing a flag.

The failing op is x500_core_jax.py:375
    return jnp.concatenate([v, acc, qdot], -1)      # widths 3, 3, 4
with v = x[..., 3:6] -> f64[8,3], matching the reported
    slice={[0:8], [3:6]} ... f64[8,3]
"""
import os
import subprocess
import sys

CASES = {

"A. bare concatenate [3,3,4], f64, batch 8": """
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
@jax.jit
def f(x):
    v, acc, q = x[..., 3:6], x[..., 0:3] * 2.0, x[..., 6:10]
    return jnp.concatenate([v, acc, q], -1)
print(f(jnp.ones((8, 10))).shape)
""",

"B. same, inside lax.scan (as in _solve)": """
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
def body(c, _):
    v, acc, q = c[..., 3:6], c[..., 0:3] * 2.0, c[..., 6:10]
    return jnp.concatenate([v, acc, q], -1), 0.0
@jax.jit
def f(x):
    return jax.lax.scan(body, x, None, length=4)[0]
print(f(jnp.ones((8, 10))).shape)
""",

"C. same, inside while_loop (error said while/body)": """
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
@jax.jit
def f(x):
    def cond(s): return s[0] < 4
    def body(s):
        i, c = s
        v, acc, q = c[..., 3:6], c[..., 0:3] * 2.0, c[..., 6:10]
        return (i + 1, jnp.concatenate([v, acc, q], -1))
    return jax.lax.while_loop(cond, body, (0, x))[1]
print(f(jnp.ones((8, 10))).shape)
""",

"D. FIX: .at[].set() instead of concatenate": """
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
@jax.jit
def f(x):
    v, acc, q = x[..., 3:6], x[..., 0:3] * 2.0, x[..., 6:10]
    out = jnp.zeros_like(x)
    out = out.at[..., 0:3].set(v)
    out = out.at[..., 3:6].set(acc)
    out = out.at[..., 6:10].set(q)
    return out
print(f(jnp.ones((8, 10))).shape)
""",

"E. FIX: .at[].set() inside while_loop": """
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
@jax.jit
def f(x):
    def cond(s): return s[0] < 4
    def body(s):
        i, c = s
        v, acc, q = c[..., 3:6], c[..., 0:3] * 2.0, c[..., 6:10]
        out = jnp.zeros_like(c)
        out = out.at[..., 0:3].set(v)
        out = out.at[..., 3:6].set(acc)
        out = out.at[..., 6:10].set(q)
        return (i + 1, out)
    return jax.lax.while_loop(cond, body, (0, x))[1]
print(f(jnp.ones((8, 10))).shape)
""",

"F. float32 instead of float64": """
import jax, jax.numpy as jnp
@jax.jit
def f(x):
    v, acc, q = x[..., 3:6], x[..., 0:3] * 2.0, x[..., 6:10]
    return jnp.concatenate([v, acc, q], -1)
print(f(jnp.ones((8, 10), jnp.float32)).shape)
""",
}

REAL = """
import sys
sys.path.insert(0, "..") ; sys.path.insert(0, ".")
import _nbinit            # noqa: F401
import x500_core_jax as X
import study_prelude as S
env = S.ev_env("circle", spec=S.nominal_spec(speed=(1.0, 1.5)), seed=202)
c = X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"])
print("SOLVE OK rmse=%.6f" % X.stats(X.rollout_eval(env, c, 20, warmup=5))["rmse"])
"""


def run(code, env_extra=None, prefix=()):
    env = dict(os.environ)
    env.update(env_extra or {})
    try:
        p = subprocess.run(list(prefix) + [sys.executable, "-c", code],
                           env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError as e:
        return None, f"cannot run: {e}"
    if p.returncode == 0:
        return True, ""
    why = [l for l in p.stderr.splitlines() if "FAILED_PRECONDITION" in l]
    if not why:
        why = [l for l in p.stderr.splitlines() if "Error" in l]
    return False, (why[-1][:130] if why else "see stderr")


def main():
    import shutil
    print("=== minimal patterns ===")
    res = {}
    for name, code in CASES.items():
        ok, why = run(code)
        res[name] = ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"       {why}")

    print("\n=== core count (this box has "
          f"{os.cpu_count()} logical cores) ===")
    taskset = shutil.which("taskset")
    if not taskset:
        print("  taskset not available, skipped")
    else:
        for n in (1, 2, 4, 8):
            ok, why = run(REAL, prefix=(taskset, "-c", f"0-{n-1}"))
            print(f"[{'PASS' if ok else 'FAIL'}] real solve on {n} core(s)")
            if ok:
                print(f"\n  >>> WORKAROUND: taskset -c 0-{n-1} python <notebook>.py")
                break

    print("\n=== verdict ===")
    cc = [k for k in res if k.startswith(("A", "B", "C")) and res[k] is False]
    ff = [k for k in res if k.startswith(("D", "E")) and res[k] is True]
    if cc and ff:
        print("  The concatenate pattern is the trigger AND .at[].set() avoids it.")
        print("  -> a code-level fix in the repo will work. Send me this output.")
    elif cc and not ff:
        print("  Concatenate reproduces, but .at[].set() does NOT help.")
        print("  -> needs a different fix. Send me this output.")
    elif not cc:
        print("  The minimal patterns all pass, so the trigger needs more of the")
        print("  real program. Send me this output; the core-count rows matter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
