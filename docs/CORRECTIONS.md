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


## C-6 — §11 T-12 measures the one quantity the mixer does not change

**Spec:** T-12 compares `mixer='nominal'` with `mixer='true'` under a 0.11
arm-tip payload and accepts when *"steady ‖τ_ext‖ [is] at least 5× larger under
`'nominal'`"*.

**Derivation.** In steady hover `ω = ω̇ = 0`, so (4.11) reduces to
`τ_ext = −M_τ^nom K_T Ω²`. Equilibrium about the *true* CG means
`M_τ^true K_T Ω² = 0`, so

```
τ_ext = −(M_τ^nom − M_τ^true) f = −T·(δ_y, −δ_x, 0),   δ = cg_true − cg_nom
      = −m_p g · (0.174, −0.174, 0)
```

which depends only on the CG shift and the total thrust. **The mixer does not
appear.** (4.11) references the nominal allocation by construction — that is
what makes a payload visible at all, and what §9.15's parameter-free check
relies on — so `τ_ext` is the same whichever mixer flies the aircraft.

**Measurement.** `asym` payload, hover, averaged over the last 4 s of 20 s:

| f | mixer | ‖I_ω‖ | ‖τ_cmd‖ [N·m] | PWM spread | ‖τ_ext‖ [N·m] | tilt |
|---|---|---|---|---|---|---|
| 0.07 | nominal | 1.518 | 0.4946 | 0.1417 | **0.3556** | 6.03° |
| 0.07 | true | 0.000 | 0.0000 | 0.1044 | **0.3487** | 0.00° |
| 0.11 | nominal | 2.526 | 1.2988 | 0.2455 | **0.5941** | 21.57° |
| 0.11 | true | 0.000 | 0.0000 | 0.1615 | **0.5480** | 0.00° |

‖τ_ext‖ differs by 2–8 %, never 5×. What differs by an *unbounded* factor is the
rate-loop effort: under the true mixer the rate loop does literally nothing —
`I_ω = 0`, `τ_cmd = 0`, tilt `0.00°` — because the off-centre load is allocated
away before it is felt, exactly as §3.3 warns.

**Implemented:** T-12 asserts on ‖τ_cmd‖ (equivalently `J·K_i·I_ω`), which is the
quantity the design decision is about, and *reports* ‖τ_ext‖ under both mixers as
the quantity that does not discriminate. This matters beyond the test: someone
reconciling the literal T-12 would be pushed to make `external_wrench` use the
mixer's belief instead of the nominal allocation, which would make every payload
invisible under `mixer='true'` and is the same substitution §9.4 check 3 warns
about one level up.

**Confirmed in passing:** §6.2's justification for the PWM block. A constant
`τ_x = 0.20 N·m` applied to the rate loop alone:

| t [s] | ‖ω‖ [rad/s] | PWM spread | I_ω,x |
|---|---|---|---|
| 5 | 1.46e-1 | 0.0437 | −1.588 |
| 20 | 2.00e-3 | 0.0437 | −2.091 |
| 80 | **7.15e-11** | **0.0437** | −2.098 |

The rate error goes to zero; the mixer spread holds forever; `I_ω` converges to
the predicted `τ/(J·K_i) = 2.098`. *State and setpoint alone cannot resolve a
standing moment; the actuator commands can.* (The spec's "≈5e-4 rad/s after 5 s"
is reached at ~15 s here, the integral time constant being ~4 s; the asymptote
and the conclusion are as stated.)


## C-7 — §5.9's cost-map range makes the learned controller inert

**Spec:** §5.9 gives `diag` as `S = diag(Q_LO + (Q_HI − Q_LO)·σ(z))` and notes
that this "starts at the sigmoid mid-range ≈ 5×10⁴", against learned matrices
that "span ~5 orders of magnitude".

**Why it fails.** Take that literally — a *linear* map whose mid-point is 5×10⁴ —
and compare it with the terminal matrix the same scheme uses. Measured at init:

```
S diagonal, e-block   [3.8e4 .. 6.6e4]      S diagonal, u-block  [5.2e4 .. 6.7e4]
P_term (Riccati) diag [1.8 .. 82.3]         u-block / P_term,pos  =  660x
```

At `N = 1` the current error `e₀` is *fixed*, so the only free variable in (5.2)
is `δu`. A control weight 660× the terminal cost therefore pins `δu` at zero:
measured `|δu| = 1.9e-5`. The AC-MPC controller silently degenerates to pure
feed-forward `u_ref`, the policy output stops affecting the command, the gradient
vanishes, and **every row of every sweep in Notebook 3 comes out identical** —
representation, exploration σ, GAE λ, PPO vs TRPO, and all three seeds agreed to
four decimals, with a "seed spread" of 0.0001 m. Nothing raises an exception.

A linear map over five decades is also a poor use of the policy's resolution:
90 % of its output range lies inside the top decade.

**Implemented:** the same five decades, mapped **logarithmically**,

```
S_ii = Q_LO · (Q_HI/Q_LO)^σ(z),   Q_LO = 1e-2,  Q_HI = 1e3
```

so `σ(z) = 0.5` lands on the geometric mean `√(Q_LO·Q_HI) = 3.16`, commensurate
with the hand weights (`Q_HAND` 0.5–2, `R_HAND` 0.5–1) and with the terminal
matrix (1.8–82). After the change, at init:

| rep | S diagonal | \|δu\| | sensitivity to the cost map |
|---|---|---|---|
| diag | 3.6e-1 … 3.5e1 | 0.293 | 1.161 |
| chol | 1.2e-1 … 4.8e0 | 0.549 | 0.536 |
| full | 8.5e-1 … 4.9e0 | 0.377 | 0.642 |

The §5.9 initialisation **asymmetry is preserved** and is still worth recording:
`chol`/`full` start at `A ≈ 0`, so `S ≈ Q_LO·I` and the quadratic term is
effectively absent, while `diag` starts at the geometric mean. They still do not
start from comparable places, and that still predicts the richer forms need
longer rather than being incapable.

**Related:** the map existed twice — once in `costmap_apply` for inference and
once in `_costmap_from_z` for training. Two copies that must agree, with nothing
checking that they do, means the policy can optimise one cost while the deployed
controller solves another. They are now one function, `costmap_from_z`, called
by both.

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

**N-6 — `check_moderate` containment is not subset containment.** §7.2 says the
moderated suite is *"a strict subset"* of the raw one. That holds only for a
band straddling the nominal 1. Every S3 multiplicative band lies entirely
*above* 1 (drag 2.20–3.00, lag 3.50–5.00), so contracting toward 1 moves it
**outside** the raw band. The invariants (7.1) actually guarantees, and which
`check_moderate` asserts, are: each endpoint stays on its own side of 1 (side
preservation, which is what "preserves the shape" means); the band is no wider;
and it lies inside `hull(raw band, {1})`. A naive subset test fails on a
correct moderation.

**N-7 — the raw `asym` top bracket is not flyable, and this is measurable.**
§8.5 says the raw brackets "sit against the actuator limit" and instructs
moderation to 40 %. Quantified: holding the standing moment of an arm-tip
payload needs rate-loop integrator authority

```
tau_needed = 0.174 · f · m · g       vs.   tau_available = J_xx · K_i,x · I_lim
                                                         = 0.023831 · 4 · 3
                                                         = 0.28597 N·m
```

so the largest trimmable mass fraction is **f = 0.0812**. The §6.1 `asym`
levels are 0, 0.04, 0.07, 0.11 — the top one exceeds it. Measured, flying a
hover reference for 30 s:

| f | τ_y vs (9.7) | F_z vs (9.7) | ‖I_ω‖ | tilt | pos err |
|---|---|---|---|---|---|
| 0.04 | 1.2 % | 0.1 % | 1.48 | 0.16° | 0.011 m |
| 0.044 | 7.2 % | 0.7 % | 1.63 | 0.97° | 0.012 m |
| 0.11 | 5–86 % (no steady state) | up to 165 % | **3.00, pinned** | **48°** | 1.20 m |

`central` matches (9.7) to 0.00 % at every level, which is what confirms the
wrench-truth computation itself. T-9 is therefore asserted on `central` at all
levels and on `asym` below the trim limit; the failure above it is reported as
a result in Notebook 5 rather than hidden by loosening the tolerance.
