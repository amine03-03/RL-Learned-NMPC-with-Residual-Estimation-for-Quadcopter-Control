# %% [markdown]
# # Notebook 7 — flight visualisation
#
# Six clips, same path, same condition, so the only variable is the controller:
# LQR, NMPC $N{=}1$, LLTC $N{=}1$, AC-MPC $N{=}1$, Adaptive AC-MPC $N{=}1$, and
# Adaptive + RDP on an **offset CG** (`asym` at 40 % of the top bracket, with the
# estimator attached so `d_channel()` returns the prediction).
#
# **Nothing here aborts on a bad result.** A missing checkpoint skips its clip
# and says so. A diverging controller is still filmed and listed in a "did not
# hold the path" table. The only hard failure is the *pilot* missing the path,
# behind `STRICT_START`.
#
# **Produces.** `videos/*.mp4` + manifest, `common/nb7_videos.csv`,
# `common/nb7_summary.json`, figures F25–F26.

# %%
import _nbinit  # noqa: F401
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import adaptive_core_jax as A
import x500_core_jax as X
import study_prelude as S
import study_moderate as M
import viz as V

S.header("Notebook 7 - flight visualisation")
X.style()
FIG = V.figures_dir()

STRICT_START = False
PATH = "circle"
R_FIXED = 1.2
SPEED = 1.2
SETTLE = 200
LAPS = 1.0
NIT = S.CFG["ilqr"]
SPEC = S.nominal_spec(speed=(SPEED, SPEED))

print(f"  archived {V.archive_previous()} pre-existing file(s) to videos/_previous/")

# %% [markdown]
# ## The pilot
#
# > The pilot must be a **fixed reference controller, not the controller under
# > test**. Using the controller under test is circular: one that cannot hold the
# > path drives the error *up* during the warm-up, so it can never be made to
# > start on the path, and its clip then shows divergence from wherever it
# > drifted to. A fixed pilot also gives every clip in a comparison an identical
# > initial state.
#
# Pilot = NMPC $N=1$.

# %%
S.section(1, "controllers", "assemble the clip list; a missing checkpoint skips, "
          "it does not abort")

PILOT = lambda: X.make_nmpc_ctrl(N=1, n_iter=NIT)
CLIPS, SKIPPED = [], []
CLIPS.append(("LQR", "LQR", lambda: X.make_lqr_ctrl(), None))
CLIPS.append(("NMPC1", "NMPC N=1", lambda: X.make_nmpc_ctrl(N=1, n_iter=NIT), None))

lltc = X.load_ckpt("lltc", "model.pkl")
if lltc is not None:
    CLIPS.append(("LLTC1", "LLTC N=1",
                  lambda: M.make_lltc_ctrl(X, lltc, N=1, n_iter=NIT), None))
else:
    SKIPPED.append(("LLTC N=1", "lltc/model.pkl missing -- run Notebook 2"))

acm = X.load_ckpt("acmpc", "model.pkl")
if acm is not None:
    CLIPS.append(("ACMPC1", "AC-MPC N=1",
                  lambda: X.ctrl_from_actor(acm["actor"],
                                            dict(acm["cfg"], n_diff=1)), None))
else:
    SKIPPED.append(("AC-MPC N=1", "acmpc/model.pkl missing -- run Notebook 3"))

est = X.load_ckpt("acmpc_adaptive", "estimators", "all.pkl")
ckv = None
for v in ("C", "B", "A"):
    ckv = X.load_ckpt("acmpc_adaptive", "variants", f"{v}.pkl")
    if ckv is not None:
        break
if ckv is not None and est is not None:
    p_sel, sc_sel, Hsel = A.rebuild_rdp(est[est["__selected__"]])
    mk = lambda: X.ctrl_from_actor(
        ckv["actor"], dict(ckv["cfg"], n_diff=1),
        dmod_fn=(lambda ev: ev.dmod()) if ckv.get("to_model") else None)
    CLIPS.append(("ADAPT1", "Adaptive AC-MPC N=1", mk, None))
    ASYM_LEVEL = A.moderate_levels(A.SCEN_LEVELS["asym"])[-1]
    CLIPS.append(("ADAPT_ASYM", f"Adaptive + RDP, offset CG (asym f={ASYM_LEVEL:g})",
                  mk, dict(scen="asym", level=ASYM_LEVEL, attach=True)))
else:
    SKIPPED.append(("Adaptive AC-MPC N=1",
                    "acmpc_adaptive checkpoints missing -- run Notebook 5"))
    SKIPPED.append(("Adaptive + RDP offset CG", "same"))

for name, why in SKIPPED:
    print(f"  SKIP  {name}: {why}")
print(f"  filming {len(CLIPS)} clip(s): {[c[1] for c in CLIPS]}")

# %% [markdown]
# ## Render
#
# Per §7.3: derive $T$ from the path period, draw the whole reference, couple
# `ep_len` to $T$, settle with the fixed pilot, and assert all comparison clips
# open in an identical state.

# %%
S.section(2, "render", "one clip per controller, identical initial state",
          produces="videos/*.mp4, common/nb7_videos.csv")

rows, DATAS, LABELS = [], [], []
start_states = {}
for name, title, mkctrl, extra in CLIPS:
    if extra is None:
        env = X.Env(1, 4242, 10, SPEC, (PATH,), fixed=dict(R=R_FIXED))
    else:
        env = A.AdaptEnv(1, 4242, 10, SPEC, extra["scen"], extra["level"],
                         paths=(PATH,), oracle=bool(ckv.get("to_obs", True)),
                         H=Hsel, fixed=dict(R=R_FIXED))
    T, per = V.clip_length(env, LAPS)
    # the episode must OUTLAST the clip, or a reset re-injects the offset
    env.cfg = type(env.cfg)(**{**env.cfg.__dict__, "ep_len": T + SETTLE + 200})
    env.reset()
    if extra is not None and extra.get("attach"):
        env.attach(p_sel, sc_sel)
    e_inject, e_start = M.settle_on_path(env, PILOT(), steps=SETTLE, tol=0.05,
                                         strict=STRICT_START, verbose=True,
                                         label=title)
    start_states[name] = np.asarray(env.state[0]).copy()
    row, data = V.make_clip(env, mkctrl(), name, title, SPEC, T=T, laps=LAPS,
                            stride=2)
    row.update(e_inject=e_inject, e_start=e_start)
    rows.append(row); DATAS.append(data); LABELS.append(title)
    print(f"    {title}: T={T} frames ({T*X.P.dt_c:.1f} s, "
          f"{row['laps']:.2f} laps), RMSE {row['rmse']:.4f} m, "
          f"respawns {row['respawns']}")

MAN = pd.DataFrame(rows)
S.table(MAN[["name", "title", "T", "period", "laps", "theta_span", "rmse",
             "maxerr", "e_inject", "e_start", "respawns"]],
        "nb7 video manifest",
        note="clip RMSE is over the WHOLE clip, consistently with maxerr",
        csv=("common", "nb7_videos.csv"))

# identical initial state across every comparison clip
base = start_states[CLIPS[0][0]]
worst = 0.0
for name, title, _, extra in CLIPS:
    if extra is not None:
        continue                       # the offset-CG clip is a different plant
    worst = max(worst, float(np.abs(start_states[name] - base).max()))
print(f"\n  comparison clips open in an identical state to {worst:.2e} "
      f"(requirement 1e-6): {'OK' if worst < 1e-6 else 'FAILED'}")
assert worst < 1e-6, "comparison clips do not start from the same state"
assert (MAN.respawns == 0).all(), "a clip respawned: it is not one continuous flight"
assert (MAN.theta_span >= 2 * np.pi - 1e-2).all() or PATH in ("hover", "step"), \
    "a clip does not span a full path period"
V.assert_manifest(MAN)
print("  manifest is authoritative and the video directory matches it exactly.")

# %% [markdown]
# ## Did they hold the path?

# %%
S.section(3, "verdicts", "nothing aborts on a bad result")
HOLD = MAN[["title", "rmse", "maxerr"]].copy()
HOLD["held_the_path"] = HOLD.rmse < 0.5
S.table(HOLD, "did not hold the path (held_the_path = False)",
        csv=("common", "nb7_hold.csv"))
bad = HOLD[~HOLD.held_the_path]
if len(bad):
    print(f"  {len(bad)} controller(s) did not hold the path; they are filmed and "
          f"listed, not dropped:")
    for _, r in bad.iterrows():
        print(f"    {r.title}: RMSE {r.rmse:.3f} m")
    hand = HOLD[HOLD.title.str.startswith(("LQR", "NMPC"))]
    learned = HOLD[~HOLD.title.str.startswith(("LQR", "NMPC"))]
    if len(hand) and len(learned) and hand.held_the_path.all() \
            and not learned.held_the_path.any():
        print("!" * 78)
        print("!! The split falls exactly along the training axis: every "
              "hand-built controller held the path and every learned one did "
              f"not.  That is the undertrained-policy symptom of scale table "
              f"X500_SCALE={S.SCALE!r}, not a controller-class result.")
        print("!! Re-run the training notebooks at X500_SCALE=full before "
              "reading these clips as a comparison.")
        print("!" * 78)
else:
    print("  every filmed controller held the path (RMSE < 0.5 m).")

# %% [markdown]
# ## F25 contact sheet and F26 ground tracks

# %%
S.section(4, "figures", "drawn FROM THE MANIFEST, never by globbing the directory")
V.contact_sheet(MAN, out=f"{FIG}/F25_contact_sheet.png")
MAN.to_csv(f"{FIG}/F25_contact_sheet.csv", index=False)
V.ground_tracks(DATAS, LABELS, out=f"{FIG}/F26_ground_tracks.png")
pd.DataFrame({f"{lb}_{ax}": d["p"][:, i] for lb, d in zip(LABELS, DATAS)
              for i, ax in enumerate("xyz")}).to_csv(
    f"{FIG}/F26_ground_tracks.csv", index=False)

X.save_json(dict(scale=S.SCALE, path=PATH, R=R_FIXED, speed=SPEED, laps=LAPS,
                 settle_steps=SETTLE, strict_start=STRICT_START,
                 clips=MAN.to_dict("records"),
                 skipped=[dict(name=n, why=w) for n, w in SKIPPED],
                 identical_start_max_delta=worst),
            "common", "nb7_summary.json")

# %% [markdown]
# ## What this notebook establishes
#
# - Six clips (or fewer, with every omission named) of the same path under the
#   same condition, so the only variable is the controller.
# - Every clip spans at least one full path period, draws the whole reference,
#   and respawned zero times — asserted, not assumed.
# - Every comparison clip opens from an identical state, reached by a **fixed**
#   pilot rather than by the controller under test.
# - A controller that diverged is still filmed and listed. If the divergence
#   splits along the training axis, the scale warning above says so.

# %%
S.checkpoint(manifest=("common", "nb7_videos.csv"),
             summary=("common", "nb7_summary.json"))
print("\nNotebook 7 complete.")
