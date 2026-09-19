#!/usr/bin/env python3
"""Bolt an asymmetric payload onto the Gazebo x500, or take it off again.

    python3 tools/payload_sdf.py --install          # add the payload
    python3 tools/payload_sdf.py --status           # is it on?
    python3 tools/payload_sdf.py --revert           # take it off

Why patch the model rather than inject a wrench at runtime?  Because the two
are not the same experiment.  A wrench applied to the base link reproduces the
payload's *weight* but not its *inertia*, and it needs the
``apply-link-wrench`` system plugin present in the world file, which the stock
PX4 worlds do not carry.  A rigidly attached link gives the real thing: the
composite mass, the shifted centre of gravity, and the parallel-axis inertia,
all handled by the physics engine, with nothing to keep in step at runtime.

The payload is a fixed joint to ``base_link`` at a body-frame (FLU) offset, so
the offset here is the same vector the scenario, the estimator and the plots
use.  Gazebo's model frame is FLU, the same convention the workspace uses
inside its boundary, so no conversion happens in this file and none should.

The edit is bracketed by marker comments and is idempotent: ``--install``
twice is one payload, and ``--revert`` restores the file byte-for-byte from
the ``.acmpc-orig`` copy taken on the first install.  A patched PX4 tree that
nobody remembers patching is a week of confusing results, so ``--status`` is
cheap and the launch file calls it.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

BEGIN = "<!-- ACMPC-PAYLOAD-BEGIN (tools/payload_sdf.py) -->"
END = "<!-- ACMPC-PAYLOAD-END -->"

#: Must match ``disturbance_manager.scenarios.S7_PAYLOAD_M / _R``.  Asserted
#: equal by ``check_demo``; a silent divergence here would make every plotted
#: truth line wrong in a way no single-sided test could see.
DEFAULT_MASS = 0.30
DEFAULT_OFFSET = (0.12, 0.06, -0.04)

CANDIDATES = (
    "Tools/simulation/gz/models/x500/model.sdf",
    "Tools/simulation/gz/models/x500_base/model.sdf",
)


def find_model(px4_dir=None, model=None):
    """Locate the x500 ``model.sdf``.  Explicit path wins, then $PX4_DIR."""
    if model:
        if not os.path.isfile(model):
            raise SystemExit(f"--model {model!r} does not exist")
        return os.path.abspath(model)
    roots = [px4_dir] if px4_dir else []
    roots += [os.environ.get("PX4_DIR"), os.environ.get("PX4_AUTOPILOT"),
              os.path.expanduser("~/PX4-Autopilot"),
              os.path.expanduser("~/px4/PX4-Autopilot"),
              "/opt/PX4-Autopilot"]
    tried = []
    for r in [x for x in roots if x]:
        for c in CANDIDATES:
            p = os.path.join(os.path.expanduser(r), c)
            tried.append(p)
            if os.path.isfile(p):
                return os.path.abspath(p)
    raise SystemExit(
        "cannot find the x500 model.sdf.  Pass --px4-dir /path/to/PX4-Autopilot "
        "or --model /path/to/model.sdf.\ntried:\n  " + "\n  ".join(tried))


def payload_xml(mass, offset, parent="base_link", size=0.06):
    """A point-like payload: a small box with the inertia of a uniform cube.

    The box's own inertia is two orders below the parallel-axis term at this
    offset (1.8e-4 against 5.4e-3 kg m^2 about z), so the payload is a point
    mass for every purpose that matters, and saying so beats pretending the
    box shape was chosen.
    """
    x, y, z = (float(v) for v in offset)
    m = float(mass)
    i = m * size ** 2 / 6.0                   # uniform cube about its own centre
    return f"""    {BEGIN}
    <link name="acmpc_payload">
      <pose relative_to="{parent}">{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
      <inertial>
        <mass>{m:.6f}</mass>
        <inertia>
          <ixx>{i:.9f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{i:.9f}</iyy><iyz>0</iyz>
          <izz>{i:.9f}</izz>
        </inertia>
      </inertial>
      <visual name="acmpc_payload_visual">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <material>
          <ambient>0.85 0.15 0.1 1</ambient>
          <diffuse>0.85 0.15 0.1 1</diffuse>
        </material>
      </visual>
      <collision name="acmpc_payload_collision">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
      </collision>
    </link>
    <joint name="acmpc_payload_joint" type="fixed">
      <parent>{parent}</parent>
      <child>acmpc_payload</child>
    </joint>
    {END}
"""


def is_installed(text):
    return BEGIN in text


def strip(text):
    """Remove the bracketed block, leaving everything else untouched."""
    return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "",
                  text, flags=re.S).replace("    \n", "\n")


def install(path, mass, offset, parent="base_link"):
    text = open(path).read()
    orig = path + ".acmpc-orig"
    if not os.path.exists(orig):
        shutil.copy2(path, orig)
    text = strip(text) if is_installed(text) else text
    if f'<link name="{parent}"' not in text and f"<link name='{parent}'" not in text:
        links = re.findall(r"<link\s+name=['\"]([^'\"]+)['\"]", text)
        raise SystemExit(
            f"no link named {parent!r} in {path}.\nlinks present: {links}\n"
            f"pass --parent with the right one (the x500 base is usually "
            f"'base_link').")
    # insert just before the closing </model>
    k = text.rfind("</model>")
    if k < 0:
        raise SystemExit(f"{path} has no </model>; is it an SDF model file?")
    out = text[:k] + payload_xml(mass, offset, parent) + text[k:]
    open(path, "w").write(out)
    return orig


def revert(path):
    orig = path + ".acmpc-orig"
    if os.path.exists(orig):
        shutil.copy2(orig, path)
        os.remove(orig)
        return "restored from .acmpc-orig"
    text = open(path).read()
    if not is_installed(text):
        return "nothing to revert"
    open(path, "w").write(strip(text))
    return "block removed (no .acmpc-orig was present)"


def summary(mass, offset):
    """The numbers the run config should record, printed so they are visible."""
    import numpy as np
    m_n, g = 2.0643076923076924, 9.8066
    r = np.asarray(offset, float)
    m = float(mass)
    dJ = m * (float(r @ r) * np.eye(3) - np.outer(r, r))
    return (f"  payload            m_p = {m:.4f} kg  at r_p = "
            f"({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}) m (FLU)\n"
            f"  mass fraction      {100 * m / m_n:.2f} %  of the nominal {m_n:.4f} kg\n"
            f"  composite CG shift {1e3 * m / (m_n + m) * r[0]:+.2f}, "
            f"{1e3 * m / (m_n + m) * r[1]:+.2f}, "
            f"{1e3 * m / (m_n + m) * r[2]:+.2f} mm\n"
            f"  residual F_z       {-m * g:+.4f} N\n"
            f"  residual tau_body  ({-m * g * r[1]:+.4f}, {m * g * r[0]:+.4f}, "
            f"+0.0000) N m\n"
            f"  parallel-axis dJ   diag {np.diag(dJ).round(6).tolist()} kg m^2")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--install", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--px4-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--parent", default="base_link")
    ap.add_argument("--mass", type=float, default=DEFAULT_MASS)
    ap.add_argument("--offset", type=float, nargs=3, default=list(DEFAULT_OFFSET),
                    metavar=("X", "Y", "Z"))
    a = ap.parse_args(argv)
    path = find_model(a.px4_dir, a.model)

    if a.status:
        on = is_installed(open(path).read())
        print(f"model: {path}\npayload: {'INSTALLED' if on else 'not installed'}")
        if on:
            m = re.search(r"<mass>([\d.eE+-]+)</mass>\s*<inertia>",
                          open(path).read()[open(path).read().find(BEGIN):])
            pose = re.search(re.escape(BEGIN) + r".*?<pose[^>]*>([^<]+)<",
                             open(path).read(), re.S)
            print(f"  mass  {m.group(1) if m else '?'}")
            print(f"  pose  {pose.group(1).strip() if pose else '?'}")
        return 0

    if a.revert:
        print(f"model: {path}\n{revert(path)}")
        return 0

    orig = install(path, a.mass, a.offset, a.parent)
    print(f"model:  {path}\nbackup: {orig}\npayload INSTALLED\n")
    print(summary(a.mass, a.offset))
    print("\nGazebo caches models; restart SITL for this to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
