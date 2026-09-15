"""Watchdog and timing for the RDP node (§9.3).

**The RDP node must never block the ACMPC loop.**  On inference failure or
timeout the estimate falls back to zero and a fault flag is set, logged and
counted.  A fallback is a *result*, not an error to be swallowed: a run whose
fault count is nonzero has to say so in its metrics.
"""
from __future__ import annotations

import time

import numpy as np

DEADLINE_MS = 20.0       # (9.4) T_RDP + T_ACMPC + T_ROS2 < 20 ms


class Watchdog:
    """Run a callable under a soft time budget, falling back to zeros."""

    def __init__(self, budget_ms=8.0, out_dim=6):
        self.budget_ms = float(budget_ms)
        self.out_dim = int(out_dim)
        self.faults = 0
        self.timeouts = 0
        self.calls = 0
        self.last_fault = ""

    def __call__(self, fn, *args, **kw):
        self.calls += 1
        t0 = time.perf_counter()
        try:
            y = np.asarray(fn(*args, **kw), dtype=float).reshape(-1)
            dt = 1e3 * (time.perf_counter() - t0)
            if y.size != self.out_dim or not np.all(np.isfinite(y)):
                raise ValueError(f"bad estimator output shape/finiteness: {y.shape}")
            if dt > self.budget_ms:
                # the answer is late but valid: use it, and count the overrun so
                # the deadline-miss rate is measured rather than assumed
                self.timeouts += 1
            return y, dt, False
        except Exception as ex:                       # noqa: BLE001
            self.faults += 1
            self.last_fault = f"{type(ex).__name__}: {ex}"
            return np.zeros(self.out_dim), 1e3 * (time.perf_counter() - t0), True

    def stats(self):
        return dict(calls=self.calls, faults=self.faults, timeouts=self.timeouts,
                    fault_rate=self.faults / max(self.calls, 1),
                    last_fault=self.last_fault)


class TimingRecorder:
    """mean / p95 / p99 / max and the deadline-miss rate, per component (§9.12)."""

    def __init__(self, deadline_ms=DEADLINE_MS):
        self.deadline_ms = float(deadline_ms)
        self._t = {}

    def add(self, label, ms):
        self._t.setdefault(label, []).append(float(ms))

    def summary(self, label=None):
        out = {}
        for k, v in self._t.items():
            if label and k != label:
                continue
            a = np.asarray(v)
            out[k] = dict(n=int(a.size), mean=float(a.mean()),
                          p95=float(np.percentile(a, 95)),
                          p99=float(np.percentile(a, 99)), max=float(a.max()),
                          miss_rate=float((a > self.deadline_ms).mean()))
        return out

    def rows(self):
        return [dict(component=k, **v) for k, v in self.summary().items()]


class CausalFilter:
    """Optional first-order causal filter on d_hat (§9.11 step 4).

    ``alpha = 0`` is a pass-through.  It is first order and causal by
    construction: no future sample is consulted, so applying it cannot violate
    §9.1.
    """

    def __init__(self, alpha=0.0, dim=6):
        self.alpha = float(alpha)
        self.y = np.zeros(int(dim))
        self._init = False

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if not self._init:
            self.y = x.copy()
            self._init = True
        else:
            self.y = self.alpha * self.y + (1.0 - self.alpha) * x
        return self.y.copy()

    def reset(self):
        self._init = False
        self.y[:] = 0.0
