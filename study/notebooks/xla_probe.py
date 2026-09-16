"""xla_probe.py -- diagnose the XLA:CPU tiling crash and find a working flag.

Symptom this exists for (seen on jax/jaxlib 0.10.2, Notebook 1 section [4]):

    jax.errors.JaxRuntimeError: FAILED_PRECONDITION: Expected the tile size of
    the concatenation dimension of operand ... f64[8,3] ... to divide the
    dimension size exactly, but got 3 % 4 != 0

That is NOT a bug in this repository.  The op is `x[..., 3:6]` in `step_c`
(x500_core_jax.py:364) -- the 3-wide velocity block -- feeding a concatenate
when the derivative vector is reassembled.  XLA's CPU backend picks a 4-wide
f64 vector tile and its emitter asserts the concatenated dimension divides
evenly by that tile.  3 % 4 != 0, so the compiler aborts.  It is a missing
edge case for non-power-of-two widths, and a 3-vector is exactly that.

The same code compiles and runs on the same jax version elsewhere, so the
trigger is environmental: either XLA_*/JAX_* variables already set in your
shell, or CPU-dependent tiling choices.  This script dumps both, then tries
each workaround in a subprocess (XLA_FLAGS is read once at backend init and
cannot be changed in-process) and tells you which one to use.

    cd study/notebooks && python xla_probe.py
"""
import os
import subprocess
import sys

# Only flags verified to be ACCEPTED by jaxlib 0.10.2; unknown flags abort the
# process and would otherwise look like a failure of the workaround itself.
# Ordered by how directly each addresses a tiling assertion in an emitter.
CASES = [
    ("baseline (no flags)",           ""),
    ("no tiling propagation",         "--xla_cpu_experimental_enable_tiling_propagation=false"),
    ("vector width 128",              "--xla_cpu_prefer_vector_width=128"),
    ("vector width 512",              "--xla_cpu_prefer_vector_width=512"),
    ("cap ISA at AVX2",               "--xla_cpu_max_isa=AVX2"),
    ("cap ISA at AVX",                "--xla_cpu_max_isa=AVX"),
    ("cap ISA at SSE4_2",             "--xla_cpu_max_isa=SSE4_2"),
    ("no thunk runtime",              "--xla_cpu_use_thunk_runtime=false"),
    ("single codegen part",           "--xla_cpu_parallel_codegen_split_count=1"),
    ("no concurrency scheduler",      "--xla_cpu_enable_concurrency_optimized_scheduler=false"),
    ("no fast-math",                  "--xla_cpu_enable_fast_math=false"),
    ("no multithread eigen",          "--xla_cpu_multi_thread_eigen=false"),
    ("tiling off + AVX2",             "--xla_cpu_experimental_enable_tiling_propagation=false "
                                      "--xla_cpu_max_isa=AVX2"),
    ("tiling off + width 128",        "--xla_cpu_experimental_enable_tiling_propagation=false "
                                      "--xla_cpu_prefer_vector_width=128"),
]

# The smallest thing that reproduces nb1 [4]: one NMPC solve through _solve.
CHILD = """
import sys
sys.path.insert(0, "..") ; sys.path.insert(0, ".")
import _nbinit            # noqa: F401  -- enables float64
import x500_core_jax as X
import study_prelude as S
env = S.ev_env("circle", spec=S.nominal_spec(speed=(1.0, 1.5)), seed=202)
c = X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"])
r = X.rollout_eval(env, c, 20, warmup=5)
print("SOLVE OK rmse=%.6f" % X.stats(r)["rmse"])
"""


def diagnostics():
    """Everything that can differ between two machines on the SAME jax build --
    which is the situation here, so the cause is somewhere in this output."""
    print("--- environment ---")
    shown = False
    for k in sorted(os.environ):
        if k.startswith(("XLA", "JAX", "LIBTPU", "TF_", "OMP_", "MKL_")):
            print(f"  {k}={os.environ[k]}")
            shown = True
    if not shown:
        print("  (no XLA_*/JAX_*/OMP_*/MKL_* variables set)")
    print("--- cpu ---")
    try:
        info = open("/proc/cpuinfo").read()
        model = [l for l in info.splitlines() if "model name" in l]
        print("  " + (model[0].split(":", 1)[1].strip() if model else "unknown"))
        flags = set(info.split("flags")[1].split("\n")[0].split())
        have = [f for f in ("sse4_2", "avx", "avx2", "fma", "avx512f",
                            "avx512dq", "avx512vl") if f in flags]
        print("  vector ISA:", " ".join(have) or "none detected")
        print("  cores:", os.cpu_count())
    except Exception as e:                                   # pragma: no cover
        print("  could not read /proc/cpuinfo:", e)
    print()


def run(flags):
    env = dict(os.environ)
    if flags:
        env["XLA_FLAGS"] = flags + " " + env.get("XLA_FLAGS", "")
    else:
        env.pop("XLA_FLAGS", None)
    try:
        p = subprocess.run([sys.executable, "-c", CHILD], env=env,
                           capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "exceeded 30 min"
    if "SOLVE OK" in p.stdout:
        return "PASS", ""
    if "Unknown flag in XLA_FLAGS" in p.stderr:
        return "N/A", "flag not in this jaxlib build"
    why = [l for l in p.stderr.splitlines()
           if "FAILED_PRECONDITION" in l or "Error" in l]
    return "FAIL", (why[-1][:150] if why else p.stderr.strip().splitlines()[-1][:150]
                    if p.stderr.strip() else "no output")


def main():
    print(f"python  {sys.version.split()[0]}")
    try:
        import jax
        import jaxlib
        print(f"jax     {jax.__version__}\njaxlib  {jaxlib.__version__}")
    except Exception as e:
        print("jax import failed:", e)
    print()
    diagnostics()

    results = []
    for name, flags in CASES:
        status, why = run(flags)
        results.append((name, flags, status))
        print(f"[{status:7s}] {name:28s} {flags}")
        if why and status not in ("PASS",):
            print(f"            {why}")
        if name.startswith("baseline") and status == "PASS":
            print("\nBaseline PASSES here -- this machine does not reproduce the"
                  " bug.\nRun this on the machine that fails and send the table.")
            return 0

    print()
    winners = [f for _, f, s in results if s == "PASS" and f]
    if not winners:
        print("None of these worked. Send me this whole table and I will dig "
              "further -- the diagnostics above narrow it a lot.")
        return 1
    w = winners[0]
    print(f"USE THIS:  XLA_FLAGS=\"{w}\"")
    print()
    print("For one notebook:")
    print(f'  XLA_FLAGS="{w}" python nb1_control_problem.py')
    print("For the whole study (export it so it sticks for the session):")
    print(f'  export XLA_FLAGS="{w}"')
    print('  for n in nb?_*.py; do python "$n" || break; done')
    if len(winners) > 1:
        print(f"\n({len(winners)} options worked; the one above is the least "
              f"invasive. Others: {winners[1:3]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
