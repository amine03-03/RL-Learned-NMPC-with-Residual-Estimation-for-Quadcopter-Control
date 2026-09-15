"""viz.py -- flight recording, rendering and the video manifest.  Implements §7.3.

Three properties, each of which is a trap:

1. **The clip must span at least one full path period.**  A fixed step count
   cannot achieve this, because omega depends on the sampled radius through
   (4.6).
2. **Draw the WHOLE reference, not the flown fragment.**  Otherwise the viewer
   sees an arc and cannot tell tracking from drift.
3. **The episode must outlast the clip**, or a reset mid-clip re-injects the
   initial offset.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

import x500_core_jax as X

MAX_FRAMES = 3000


def videos_dir():
    d = X.apath("videos")
    os.makedirs(d, exist_ok=True)
    return d


def figures_dir():
    d = X.apath("figures")
    os.makedirs(d, exist_ok=True)
    return d


def describe_disturbance(spec):
    """A one-line human description of an evaluation spec."""
    if spec is None:
        return "nominal"
    bits = []
    for k in ("wind", "speed"):
        if k in spec:
            lo, hi = spec[k]
            bits.append(f"{k} {lo:g}-{hi:g}")
    lam = [k for k in spec if k in ("m", "D", "tau", "T", "Kw", "J")
           and spec[k] != (1.0, 1.0)]
    if lam:
        bits.append("scales " + ",".join(sorted(lam)))
    return "; ".join(bits) if bits else "nominal"


def path_period(env, i=0):
    """2*pi/omega, or None for a path with no period (hover/step/helix)."""
    kind = int(np.asarray(env.ep["kind"])[i])
    if X.PATHS_ALL[kind] in ("hover", "step", "helix"):
        return None
    w = float(np.asarray(env.ep["omega"])[i])
    return None if w <= 0 else 2 * np.pi / w


def clip_length(env, laps=1.0, i=0, default=400):
    """Derive T from the path period, clamped to MAX_FRAMES with a warning."""
    per = path_period(env, i)
    if per is None:
        return default, per
    T = int(np.ceil(laps * per / X.P.dt_c))
    if T > MAX_FRAMES:
        print(f"  !! clip clamped: {T} frames needed for {laps} lap(s) of a "
              f"{per:.2f} s period, capping at {MAX_FRAMES}; the clip will NOT "
              f"span a full period.")
        T = MAX_FRAMES
    return T, per


def full_reference(env, i=0, n=1200):
    """RFULL: the whole closed reference curve, sampled over one period from t=0.

    Plotting the flown fragment instead is the difference between a figure that
    shows tracking and one that shows an arc.
    """
    per = path_period(env, i)
    span = per if per is not None else 8.0
    ts = np.linspace(0.0, span, n)
    ep1 = {k: (v[i:i + 1] if hasattr(v, "shape") and v.ndim >= 1 else v)
           for k, v in env.ep.items()}
    import jax.numpy as jnp
    out = []
    for t in ts:
        p, _, _ = X._ref_pva(ep1, jnp.asarray([t]))
        out.append(np.asarray(p)[0])
    return np.array(out)


def fly(env, ctrl, T, stride=2, i=0):
    """Record a single continuous flight.  ``respawns`` must be 0 for a clip."""
    r0 = env.respawns
    env.no_respawn = True
    res = X.rollout_eval(env, ctrl, T, record=True)
    env.no_respawn = False
    rec = res["rec"]
    data = dict(
        p=np.array([f["p"][i] for f in rec]), pr=np.array([f["pr"][i] for f in rec]),
        q=np.array([f["q"][i] for f in rec]), u=np.array([f["u"][i] for f in rec]),
        t=np.array([f["t"][i] for f in rec]), om=np.array([f["om"][i] for f in rec]),
        RFULL=full_reference(env, i), stride=stride,
        respawns=env.respawns - r0)
    err = np.linalg.norm(data["p"] - data["pr"], axis=1)
    data["err"] = err
    data["rmse"] = float(np.sqrt((err ** 2).mean()))     # over the WHOLE clip
    data["maxerr"] = float(err.max())
    th = np.unwrap(np.arctan2(data["p"][:, 1] - np.mean(data["RFULL"][:, 1]),
                              data["p"][:, 0] - np.mean(data["RFULL"][:, 0])))
    data["theta_span"] = float(abs(th[-1] - th[0]))
    return data


def render_flight(data, title="", wind=None, out=None, show_ref=True):
    """3-D trajectory + error/thrust traces + a wind arrow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
    X.style()
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 3, (1, 2), projection="3d")
    R = data["RFULL"]
    if show_ref:
        ax.plot(R[:, 0], R[:, 1], R[:, 2], "--", color="0.45", lw=1.2,
                label="reference (full period)")
    ax.plot(data["p"][:, 0], data["p"][:, 1], data["p"][:, 2], "-", color="C0",
            lw=1.8, label="flown")
    ax.scatter(*data["p"][0], color="C2", s=25, marker="o", label="start")
    lo, hi = R.min(0), R.max(0)                        # frame the box on RFULL
    ctr, half = (lo + hi) / 2, max((hi - lo).max(), 1.0) / 2 * 1.25
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(max(0, ctr[2] - half), ctr[2] + half)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    if wind is not None and np.linalg.norm(wind) > 1e-6:
        w = np.asarray(wind) / max(np.linalg.norm(wind), 1e-9) * half * 0.5
        ax.quiver(ctr[0], ctr[1], ctr[2] + half * 0.8, w[0], w[1], w[2],
                  color="C3", lw=2, label=f"wind {np.linalg.norm(wind):.1f} m/s")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(title, fontsize=9)

    ax2 = fig.add_subplot(2, 3, 3)
    ax2.plot(data["t"], data["err"], color="C3")
    ax2.axhline(data["rmse"], ls=":", color="0.4")
    ax2.set_ylabel("|p - p_ref| [m]"); ax2.set_xlabel("")
    ax2.set_title(f"RMSE {data['rmse']:.3f} m   max {data['maxerr']:.3f} m", fontsize=8)
    ax3 = fig.add_subplot(2, 3, 6)
    ax3.plot(data["t"], data["u"][:, 0], color="C1")
    ax3.axhline(1.0, ls="--", color="C3", lw=1)
    ax3.axhline(X.U_HOVER, ls=":", color="0.4", lw=1)
    ax3.set_ylabel("collective [-]"); ax3.set_xlabel("t [s]"); ax3.set_ylim(0, 1.05)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def make_clip(env, ctrl, name, title, spec=None, T=None, laps=1.0, stride=2,
              wind=None, fps=25):
    """Render an mp4 (or a PNG fallback where no writer exists) and return the
    manifest row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
    if T is None:
        T, per = clip_length(env, laps)
    else:
        per = path_period(env)
    data = fly(env, ctrl, T, stride)
    assert data["respawns"] == 0, (
        f"clip {name!r} respawned {data['respawns']} times -- it is not one "
        f"continuous flight; raise ep_len above T + settle + margin (§7.3)")
    X.style()
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    R = data["RFULL"]
    lo, hi = R.min(0), R.max(0)
    ctr, half = (lo + hi) / 2, max((hi - lo).max(), 1.0) / 2 * 1.25
    idx = np.arange(0, len(data["p"]), stride)

    def draw(k):
        ax.clear()
        ax.plot(R[:, 0], R[:, 1], R[:, 2], "--", color="0.45", lw=1.1)
        j = idx[k]
        ax.plot(data["p"][:j + 1, 0], data["p"][:j + 1, 1], data["p"][:j + 1, 2],
                color="C0", lw=1.8)
        ax.scatter(*data["p"][j], color="C1", s=40)
        ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(max(0, ctr[2] - half), ctr[2] + half)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_title(f"{title}\nt={data['t'][j]:.2f}s  |e|={data['err'][j]:.3f} m",
                     fontsize=9)
        return ax,

    path = os.path.join(videos_dir(), f"{name}.mp4")
    wrote = None
    try:
        anim = animation.FuncAnimation(fig, draw, frames=len(idx), blit=False)
        anim.save(path, writer=animation.FFMpegWriter(fps=fps))
        wrote = path
    except Exception as ex:
        path = os.path.join(videos_dir(), f"{name}.png")
        draw(len(idx) - 1)
        fig.savefig(path, bbox_inches="tight")
        wrote = path
        print(f"  !! no mp4 writer ({type(ex).__name__}); wrote {os.path.basename(path)}")
    # contact-sheet frame at t = 1 s
    sheet = os.path.join(videos_dir(), f"{name}_t1.png")
    k1 = int(min(len(idx) - 1, max(0, round(1.0 / X.P.dt_c / stride))))
    draw(k1); fig.savefig(sheet, bbox_inches="tight")
    plt.close(fig)
    return dict(name=name, title=title, file=os.path.basename(wrote),
                sheet=os.path.basename(sheet), T=T, period=per,
                laps=(T * X.P.dt_c / per) if per else np.nan,
                theta_span=data["theta_span"], rmse=data["rmse"],
                maxerr=data["maxerr"], respawns=data["respawns"],
                disturbance=describe_disturbance(spec)), data


def archive_previous():
    """Move pre-existing clips to ``_previous/`` before rendering (§7.3)."""
    d = videos_dir()
    prev = os.path.join(d, "_previous")
    os.makedirs(prev, exist_ok=True)
    moved = 0
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            shutil.move(p, os.path.join(prev, f))
            moved += 1
    return moved


def assert_manifest(manifest):
    """The manifest is authoritative: the directory must contain exactly its
    files.  Contact sheets read the manifest, never glob the directory."""
    d = videos_dir()
    on_disk = {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}
    want = set(manifest["file"]) | set(manifest["sheet"])
    missing, extra = want - on_disk, on_disk - want
    if missing:
        raise AssertionError(f"manifest lists files not on disk: {sorted(missing)}")
    if extra:
        raise AssertionError(f"video directory holds files not in the manifest: "
                             f"{sorted(extra)}")
    return True


def contact_sheet(manifest, out=None, ncol=3):
    """F25: one frame per clip at t = 1 s, drawn **from the manifest**."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    X.style()
    rows = list(manifest.to_dict("records"))
    nr = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nr, ncol, figsize=(4.0 * ncol, 3.2 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, rows):
        p = os.path.join(videos_dir(), r["sheet"])
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        ax.set_title(f"{r['title']}\nRMSE {r['rmse']:.3f} m", fontsize=8)
        ax.axis("off")
    for ax in axes[len(rows):]:
        ax.axis("off")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def ground_tracks(datas, labels, out=None):
    """F26: overlaid ground tracks of every controller on the full reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X.style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    R = datas[0]["RFULL"]
    ax.plot(R[:, 0], R[:, 1], "--", color="0.35", lw=1.4, label="reference")
    mk = ["o", "s", "^", "v", "D", "P", "X"]
    for i, (d, lb) in enumerate(zip(datas, labels)):
        ax.plot(d["p"][:, 0], d["p"][:, 1], lw=1.4, label=lb,
                marker=mk[i % len(mk)], markevery=max(len(d["p"]) // 12, 1), ms=4)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    ax.legend(fontsize=7)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig
