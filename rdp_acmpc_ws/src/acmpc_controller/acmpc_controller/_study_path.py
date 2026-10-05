"""Put ``study/`` on ``sys.path`` so ``x500_core_jax`` imports.

The controller deliberately uses the study's model, constants and reference
functions instead of a second copy (see docs/ROS2_WORKSPACE.md §1).  The path
is resolved relative to this file, which works from the source tree and from a
``colcon build --symlink-install``; otherwise put ``study/`` on ``PYTHONPATH``.
"""
import os
import sys

STUDY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..", "..", "study"))
if os.path.isdir(STUDY) and STUDY not in sys.path:
    sys.path.insert(0, STUDY)
