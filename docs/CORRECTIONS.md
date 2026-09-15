# Corrections to `AGENTS_SPEC_Adaptive_ACMPC.md`

The spec says (§ rules of engagement) *"if a table contradicts the surrounding
prose, the prose is wrong"* and *"never let a claim outrun a measurement"*.
Applying that to the spec itself: every constant in §2 and §4.2 was re-derived
and reproduces to the stated precision, but **five statements do not survive
checking**. Each is listed with the measurement that refutes it and the form
this repository implements. Reproduce all of them with:

```bash
python -m pytest study/tests/test_spec_corrections.py -q
```

---

## C-1 — (5.6) `A_c` attitude block has the wrong sign  *(decisive)*

**Spec:** the velocity/attitude block of the hover linearisation is `-g·Ξ`, with
`Ξ = [[0,1,0],[-1,0,0],[0,0,0]]`.

**Derivation.** With `δ = 2·q_ev` the error attitude of (4.1), `R ≈ I + [δ]ₓ` for
small tilt, so `z_b = R e₃ ≈ e₃ + δ × e₃ = e₃ + (δ_y, −δ_x, 0)`. At hover
`T/m = g`, hence `∂v̇/∂δ = g·[[0,1,0],[-1,0,0],[0,0,0]] = +g·Ξ`.

**Measurement.** `jax.jacobian` of the error dynamics (4.1)+(3.3) at the hover
trim gives exactly `+g·Ξ`. Taking the spec's sign and solving the Riccati
equation produces a gain which, applied to the true plant, has closed-loop
spectral radius **1.1807 — unstable**. The corrected sign gives 0.9386.

**Implemented:** `+g·Ξ`. `x500_core_jax.lqr_matrices()` *derives* `A_c, B_c` by
autodiff and asserts them against the corrected analytic form, so the sign
cannot regress. Test `T-5b`.

## C-2 — (5.6) `B_c` attitude block is off by a factor of two

**Spec:** `∂δ̇/∂ω̄ = ½·diag(ω_max)`.

**Derivation.** `q̇ = ½ q ⊗ [0,ω]` gives `q̇_v ≈ ½ω` at `q ≈ 1`; since
`δ = 2q_v`, `δ̇ = ω = diag(ω_max)·ω̄`. The factor ½ is already consumed by the
factor 2 in the definition of `δ`.

**Measurement.** autodiff gives `diag(10, 10, 4)`, not `(5, 5, 2)`.

**Implemented:** `diag(ω_max)`. Same test.

> C-1 and C-2 are the only two places in the spec where the `2` of
> `e₇..₉ = 2·sgn(q_w)·q_v` was dropped; everything else in §4–§5 is consistent
> with it.

## C-3 — (6.1) slung-load tension has the wrong sign

**Spec:** `F_ten = m_ℓ(g·n̂·e₃ + L‖ṅ̂‖²)`, with `F^b_ext = Rᵀ(F_ten·n̂)`.

**Derivation.** With `n̂` the unit vector from the attachment point to the load,
projecting the load's equation of motion on `n̂` gives
`T = m_ℓ(−g·(n̂·e₃) − a_q·n̂ + L‖ṅ̂‖²)`. A statically hanging load has `n̂ = −e₃`
and must give `T = +m_ℓ g`; the spec's form gives `−m_ℓ g`, i.e. a load that
pushes the airframe *up*.

**Measurement.** hanging, `m_ℓ = 0.149·m`: spec `−3.0163 N`, corrected
`+3.0163 N`, static truth `+3.0163 N`.

**Implemented:** `F_ten = m_ℓ(−g·(n̂·e₃) + L‖ṅ̂‖²)`, `F^world = F_ten·n̂`, which is
downward for a hanging load. The `a_q·n̂` coupling is dropped (it forms an
algebraic loop with the airframe acceleration); this is the spec's intent and is
declared in the docstring. Test `T-19b` checks the static tension and that the
free-pendulum period is `2π√(L/g) = 1.4187 s`.

## C-4 — §7.2 the moderated suites are not disjoint on thrust

**Spec:** *"disjoint on drag, lag, thrust and wind but overlap on mass"*.

**Measurement.** Applying (7.1) with `φ = 0.15` to the §7.2 table:

| channel | S2 moderated | S3 moderated | |
|---|---|---|---|
| wind | 0.000–2.000 | 2.500–4.000 | disjoint |
| drag `λ_D` | 0.925–1.150 | 1.180–1.300 | disjoint |
| lag `λ_τ` | 0.925–1.300 | 1.375–1.600 | disjoint |
| mass `λ_m` | 0.970–1.030 | 0.948–1.053 | **overlap** |
| thrust `λ_T` | 0.978–1.023 | 0.963–1.045 | **overlap** |
| gain `λ_Kw` | 0.940–1.060 | 0.910–1.120 | **overlap** |
| inertia `λ_J` | 0.955–1.060 | 0.925–1.120 | **overlap** |

Wherever the raw S3 band is a *superset* of the raw S2 band — which is true for
mass, thrust, `K_w` and inertia — contraction toward the common centre 1
preserves the inclusion. Only the four channels whose raw bands were already
separated stay separated.

**Implemented:** `study_moderate.disjointness(before, after)` computes the table
above from the specs rather than asserting the claim, and the notebooks print
it. The corrected sentence is: *disjoint on drag, lag and wind; overlapping on
mass, thrust, `K_w` and inertia.* Test `T-14b`.

## C-5 — §4.2 the stated consequence of the (4.5) sign slip is wrong

**Spec:** writing the second term of (4.5) as `−` *"produces `κ_a = 4.6464`
instead of 9.0779"*.

**Measurement.** The `+` sign is correct and gives `κ_a = 9.077877`, which is
what matters and which this repo reproduces to 1e-6 by dense sampling and by
central difference. But the sign-flipped variant actually gives **6.8602**, not
4.6464. The warning stands; only its illustrative number is wrong.

**Implemented:** `KAPPA_A` is *computed* by dense sampling at import and asserted
against the §4.2 table, so no variant of (4.5) can be adopted silently. Test
`T-4`.

---

## Implementation notes that are not corrections

**N-1 — (4.2) is implemented in its equivalent closed form.** `n = 2·arcsin(‖δ‖/2)`
gives `cos(n/2) = √(1−‖δ‖²/4)` and `δ̂·sin(n/2) = δ/2`, so
`q_e = [√(1−‖δ‖²/4), δ/2]`. The two agree to 1.3e-14 over 1000 random attitudes,
but the spec's form divides by `‖δ‖` and so has a **NaN gradient at zero error** —
which is precisely the state the iLQR linearises about at convergence. The
closed form is smooth there. `e_to_state` uses it.

**N-2 — `_clamp` gradient convention (§5.8).** The derivative is 1 for
`lo ≤ x ≤ hi` and 0 outside: pass-through *on* the boundary. Taking 0 there
would zero the collective column of `B` exactly when the line search parks the
input on the box, making `Q_uu` singular in that direction and blinding the
solver. Consequence for T-7: at the boundary the reference is a **one-sided**
difference taken into the feasible interior, not a central one (a central
difference across a kink returns ½ and would flag a correct implementation).

**N-3 — `edyn` takes `uref` explicitly.** The spec's signature
`edyn(e, du, xr, xr_next, d, par)` cannot recover `u_ref,k`: (4.7) fixes only the
*direction* of `a_ref + g e₃`, not its magnitude. `uref` is therefore an optional
trailing argument; when omitted it is reconstructed from `(v_r,k+1 − v_r,k)/Δt_c`,
which is first-order. `Env.ref_useq(N)` supplies the exact analytic value and
every controller that binds the environment uses it.

**N-4 — nominal vs. true `T_max`.** `λ_T` is a *plant* perturbation: the
allocator and the control model use the nominal `T_max` (`par['T_max_ctrl']`) and
only the rotors produce `λ_T`-scaled thrust (`par['KT']`). Were the allocator to
use the scaled value there would be no thrust-model error to estimate. This is
the same failure mode §9.4 check 3 warns about, one level up.

**N-5 — rate-loop integrator anti-windup.** (3.8) integrates `İ_ω = ω_c − ω_f`
and §3.3 caps `‖I_ω‖_∞ ≤ 3`. Clamping inside the RK4 derivative makes the stage
derivatives inconsistent; the state is clamped after each completed sub-step
instead, which is what a discrete autopilot does.
