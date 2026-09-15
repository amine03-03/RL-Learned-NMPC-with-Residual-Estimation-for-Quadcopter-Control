"""Shared notebook preamble: path setup and the common imports."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(STUDY)
for p in (STUDY, os.path.join(ROOT, "rdp_acmpc_ws", "src", "acmpc_controller")):
    if p not in sys.path:
        sys.path.insert(0, p)
