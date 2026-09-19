import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(STUDY)
WS = os.path.join(ROOT, "rdp_acmpc_ws")
for p in (STUDY,
          os.path.join(WS, "src", "acmpc_controller"),
          os.path.join(WS, "src", "disturbance_manager"),
          os.path.join(WS, "src", "reference_generator"),
          os.path.join(WS, "src", "rdp_estimator"),
          os.path.join(WS, "src", "visualization"),
          os.path.join(WS, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
