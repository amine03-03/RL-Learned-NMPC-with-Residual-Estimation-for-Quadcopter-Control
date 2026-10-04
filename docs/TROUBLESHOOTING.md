# Troubleshooting

## `FAILED_PRECONDITION: ... tile size ... 3 % 4 != 0` (CUDA backend)

Symptom — any notebook, as soon as an MPC solve is compiled:

```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: Expected the tile size of the
concatenation dimension of operand %slice... = f64[8,3] slice(...),
slice={[0:8], [3:6]}, metadata={op_name="jit(_solve)/.../jit(step_c)/slice"}
to divide the dimension size exactly, but got 3 % 4 != 0
```

**Status: fixed in this repository.** If you are on a current checkout you
should not see it. The rest of this section explains what it was, because the
shape of the bug makes it easy to re-introduce.

### What it was

XLA aborts while compiling a `concatenate` whose operands are 3 wide. The
emitter picks a 4-element f64 tile and asserts that the concatenated dimension
divides evenly by that tile; a 3-vector never will. It is a missing edge case
for non-power-of-two widths, not a problem with the traced program.

The trigger was `fc()` in `x500_core_jax.py`, which assembled the state
derivative as `jnp.concatenate([v, acc, qdot], -1)` with widths 3, 3, 4 —
`v = x[..., 3:6]` being the `f64[8,3]` in the message — plus the same pattern
in the quaternion term and in `step_c`.

`fc` and `step_c` now assemble their outputs with `_assemble`, which writes
blocks into a preallocated array with `.at[].set()`. That lowers to
dynamic-update-slice and never reaches the concatenate emitter. It is exactly
equivalent: verified bit-identical (`0.0`) across flat, batched and single
shapes, with and without the disturbance argument, and pinned by `test_D8_*`
in `study/tests/test_audit_corrections.py`.

Verified after the change: the full suite passes (86 tests) and Notebook 1 runs
to completion with its physics self-test green, so none of the corrected
constants moved.

**On "identical".** `fc` and `step_c` are bit-identical to the concatenate
version at equal inputs (`0.0` exactly). Closed-loop *aggregates* are not quite,
because the different lowering lets XLA fuse and reassociate downstream
arithmetic differently, and a 250-step closed loop amplifies last-bit
differences. Measured across the Notebook 1 artifacts:

| metric | worst relative change |
|---|---|
| `rmse` | 1.4e-10 |
| `rmse_iqr` | 4.5e-10 |
| `tilt`, `effort` | ~3e-9 |
| `maxerr` | 3.0e-8 |
| `smooth` | 3.7e-8 |

`maxerr` and `smooth` sit highest because they are cancellation-sensitive — a
max over a trajectory and a second difference of the control. All of it is far
below any physically meaningful scale, but it is not zero, and committed
artifacts produced before this change differ from a fresh run by these amounts.

**Do not run a single notebook to "check" things and commit the result.**
`artifacts/common/ledger.csv` is appended to across the chain; running Notebook 1
alone truncates it from six rows to one.

### It is backend-specific, and that is the trap

**This only happens on the CUDA backend.** The same code compiles fine on a
CPU-only jaxlib, on the same jax/jaxlib version. A machine with a GPU installs
a CUDA jaxlib, JAX puts the computation on the GPU by default, and the crash
appears; a CPU-only box never sees it.

Two consequences, both of which cost real time when this was first diagnosed:

- **No `--xla_cpu_*` flag can fix it.** They are silently no-ops, because the
  computation is not on the CPU backend. A flag sweep will return *byte
  identical* errors for every flag, which looks like evidence about compiler
  passes and is actually evidence that none of the flags applied.
- **Check which backend you are on before anything else.**
  `python -c "import jax; jax.print_environment_info()"` prints a `device info:`
  line. If it names a GPU, you are not on the CPU backend.

### If it comes back somewhere else

There are ~44 `jnp.concatenate` calls in the study and several have
non-power-of-two operand widths, so another one could trip the same emitter on
a code path that is not yet exercised. The fix is mechanical: replace the
concatenate with `_assemble(...)`, and add the numbers to the `test_D8_*`
checks.

### Escape hatch

If you hit a variant and need to keep working immediately, force the CPU
backend:

```bash
JAX_PLATFORMS=cpu python nb1_control_problem.py
```

This is a workaround, not a fix — it gives up the GPU, which is a large
slowdown for the training notebooks.

## Versions

There is no pinned dependency file, so a fresh virtualenv can resolve to a JAX
build that has not been exercised here. Verified: **jax 0.10.2 / jaxlib 0.10.2**,
float64 enabled, on both a CPU-only build and a CUDA build (RTX 4080 SUPER,
driver 580.178.04, CUDA 13.0).
