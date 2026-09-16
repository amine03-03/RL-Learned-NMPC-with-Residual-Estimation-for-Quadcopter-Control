# Troubleshooting

## `FAILED_PRECONDITION: ... tile size ... 3 % 4 != 0` during a notebook

Full symptom, seen on jax/jaxlib 0.10.2 in Notebook 1 section [4]:

```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: Expected the tile size of the
concatenation dimension of operand %slice... = f64[8,3] slice(...),
slice={[0:8], [3:6]}, metadata={op_name="jit(_solve)/.../jit(step_c)/slice"}
to divide the dimension size exactly, but got 3 % 4 != 0
```

**This is not a bug in this repository.** The op is `x[..., 3:6]` in `step_c`
(`x500_core_jax.py:364`) — the 3-wide velocity block — feeding a `concatenate`
when the state derivative is reassembled. XLA's CPU backend picks a 4-wide f64
vector tile (256-bit) and its emitter asserts the concatenated dimension
divides evenly by that tile. `3 % 4 != 0`, so the compiler aborts. It is a
missing edge case for non-power-of-two widths, and a 3-vector is exactly that.

It is a *compiler* precondition failure, not a shape error: nothing is wrong
with the traced program, and the identical code compiles and runs on the same
jax version on other machines. The trigger is environmental — `XLA_*`/`JAX_*`
variables already set in your shell, or CPU-dependent tiling choices.

### Quickest thing to try

```bash
XLA_FLAGS="--xla_cpu_experimental_enable_tiling_propagation=false" \
  python nb1_control_problem.py
```

That flag disables the tiling pass whose assertion is failing, and is the
least invasive of the known workarounds.

### If that does not work

```bash
cd study/notebooks && python xla_probe.py
```

`xla_probe.py` dumps the environment and CPU (the two things that can differ
between machines on an identical jax build), then reproduces the failing solve
under each candidate flag in a subprocess — `XLA_FLAGS` is read once at backend
initialisation and cannot be changed in-process — and prints the exact
`XLA_FLAGS` line to use for the whole study.

Only flags **verified to be accepted by jaxlib 0.10.2** are tried. An
unrecognised flag aborts the process, which would otherwise look identical to
the workaround failing; the probe reports those as `N/A` rather than `FAIL`.

### Applying a workaround to the whole study

```bash
export XLA_FLAGS="<the line the probe prints>"
for n in nb?_*.py; do python "$n" || break; done
```

### Note on versions

There is no pinned dependency file in this repository, so a fresh virtualenv
can resolve to a JAX build that has not been exercised here. The verified
combination is **jax 0.10.2 / jaxlib 0.10.2 on CPU with float64 enabled**.
