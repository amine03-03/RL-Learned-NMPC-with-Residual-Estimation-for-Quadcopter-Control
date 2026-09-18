"""The run that decides whether `full` is worth starting.

Medium training settings, two arms differing ONLY in EnvCfg.term, each evaluated
twice: with the 3 m respawn on (comparable with the study's tables) and off (the
true tracking error).  Answers three things at once:

  1. does the policy track at all now that steps land (G1)?
  2. does `term` change the POLICY, not just the value target (G2 end to end)?
  3. does reward improve monotonically, and where does lr settle -> what
     iteration budget `full` actually needs?

Also counts terminations, so we know G2 was exercised this time: at the small
probe size done=0 and far=0, which made the two arms bit-identical.
"""
import os, sys, time, numpy as np, jax, jax.numpy as jnp, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x500_core_jax as X

# medium scale table, verbatim from study_prelude.SCALES["medium"]
NENV, TROLL, MINIB, EPOCHS, ITERS, HID, ILQR = 128, 32, 16, 6, 80, 256, 10
SEED = 5
SPEC = X.nominal_spec(speed=(1.0, 1.5))
CFG = dict(N=1, rep="diag", n_iter=ILQR, n_diff=2, hid=HID, minib=MINIB,
           epochs=EPOCHS, sigma=0.05, algo="ppo", mpve=False, lam=0.95)


def count_terminations(term, actor, steps=200):
    """Is G2 even reachable at this rollout length?"""
    env = X.Env(NENV, SEED, 200, SPEC, ("circle",), term=term)
    ctrl = X.ctrl_from_actor(actor, dict(CFG, n_diff=1)); ctrl.bind_env(env)
    o, e, xr = env.obs(); dn = fr = 0
    for _ in range(steps):
        _, uref, _ = env.ref_now()
        u, _ = ctrl(o, e, xr, uref=uref)
        _, done, info = env.step(u)
        dn += int(jnp.sum(done)); fr += int(jnp.sum(info["far"]))
        o, e, xr = env.obs()
    return dn, fr


def ev(actor, no_respawn, T=300):
    env = X.Env(64, 900, 100000, SPEC, ("circle",))
    env.no_respawn = no_respawn
    return X.stats(X.rollout_eval(env, X.ctrl_from_actor(actor, dict(CFG, n_diff=1)),
                                  T, warmup=40))


print(f"  anchor: NMPC N=1 (hand weights), respawn on/off")
for nr in (False, True):
    env = X.Env(64, 900, 100000, SPEC, ("circle",)); env.no_respawn = nr
    c = X.make_nmpc_ctrl(N=1, n_iter=ILQR)
    st = X.stats(X.rollout_eval(env, c, 300, warmup=40))
    print(f"    no_respawn={nr!s:5s}  rmse {st['rmse']:.4f}  maxerr {st['maxerr']:.3f}"
          f"  sat {st['sat']:.4f}  bound_frac {st['bound_frac']:.4f}")

OUT = {}
for term in ("bootstrap", "cut"):
    t0 = time.time()
    env = X.Env(NENV, SEED, 200, SPEC, ("circle",), term=term)
    actor, critic, log = X.train_ppo(None, dict(CFG, dist_label=f"term={term}"),
                                     seed=SEED, iters=ITERS, T_rollout=TROLL,
                                     env=env, verbose=True)
    wall = time.time() - t0
    dn, fr = count_terminations(term, actor)
    on, off = ev(actor, False), ev(actor, True)
    tail = log.iloc[int(0.8 * len(log)):]
    OUT[term] = dict(
        term=term, wall_s=round(wall, 1),
        landed=int(log.landed.sum()), intended=ITERS * EPOCHS * MINIB,
        lr_first=log.lr.iloc[0], lr_last=log.lr.iloc[-1], lr_max=log.lr.max(),
        rew_first=log.reward.iloc[:5].mean(), rew_last=log.reward.iloc[-5:].mean(),
        sat_tail=tail.sat.mean(), sat_gate_ok=bool(tail.sat.mean() <= 0.05),
        terminations=dn, far_events=fr,
        rmse_respawn_on=on["rmse"], rmse_respawn_off=off["rmse"],
        maxerr_off=off["maxerr"], bound_frac_on=on["bound_frac"])
    log.to_csv(f"./step1_log_{term}.csv", index=False)
    print(f"\n  ---- term={term} done in {wall/60:.1f} min ----")
    for k, v in OUT[term].items():
        print(f"    {k:20s} {v}")

D = pd.DataFrame(OUT.values())
print("\n=== STEP 1 SUMMARY ===")
print(D.to_string(index=False))
D.to_csv("./step1_summary.csv", index=False)
b, c = OUT["bootstrap"], OUT["cut"]
print(f"\n  G2 exercised?  terminations bootstrap={b['terminations']} cut={c['terminations']}"
      f"  (0 in both => the two arms are identical by construction)")
print(f"  G2 effect on the POLICY:  rmse(no_respawn) bootstrap {b['rmse_respawn_off']:.3f}"
      f"  vs cut {c['rmse_respawn_off']:.3f}")
print(f"  does it track?  bootstrap rmse_off {b['rmse_respawn_off']:.3f} m against"
      f" NMPC N=1 (see anchor above)")
print(f"  sat gate:  bootstrap {b['sat_tail']:.3f} ({'PASS' if b['sat_gate_ok'] else 'FAIL'})"
      f"   cut {c['sat_tail']:.3f} ({'PASS' if c['sat_gate_ok'] else 'FAIL'})")
