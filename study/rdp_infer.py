"""rdp_infer.py -- **pure NumPy** forward pass for the exported RDP.

Shipped into the ROS 2 workspace.  It must import with neither JAX nor PyTorch
on the path: the node runs inside a 20 ms control budget and cannot afford a
framework, and the study's JAX pytrees are not a deployment format.

Parity against the JAX predictor is asserted by ``export_estimator.py`` over 64
random windows at 1e-5 (T-17); the export **fails** if parity fails, because a
silently divergent deployment model is exactly the failure this file exists to
prevent.
"""
from __future__ import annotations

import json

import numpy as np


def _sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)          # overflow-free


def _relu(x):
    return np.maximum(x, 0.0)


def _causal_conv(W, b, x, dilation):
    """x (B,T,Cin), W (k,Cin,Cout) -> (B,T,Cout), left-padded (causal)."""
    k = W.shape[0]
    pad = (k - 1) * dilation
    xp = np.pad(x, ((0, 0), (pad, 0), (0, 0)))
    T = x.shape[1]
    out = np.zeros((x.shape[0], T, W.shape[2]))
    for i in range(k):
        # tap i covers input offset i*dilation before the right edge
        sl = xp[:, i * dilation: i * dilation + T]
        out += sl @ W[i]
    return out + b


def _gru(W_i, b_i, W_h, b_h, xs):
    """xs (B,T,nin) -> hs (B,T,nh).  Same gate equations as ``gru_scan``."""
    B, T, _ = xs.shape
    nh = W_h.shape[0]
    h = np.zeros((B, nh))
    hs = np.zeros((B, T, nh))
    for t in range(T):
        gi = xs[:, t] @ W_i + b_i
        gh = h @ W_h + b_h
        r = _sigmoid(gi[:, :nh] + gh[:, :nh])
        z = _sigmoid(gi[:, nh:2 * nh] + gh[:, nh:2 * nh])
        n = np.tanh(gi[:, 2 * nh:] + r * gh[:, 2 * nh:])
        h = (1.0 - z) * n + z * h
        hs[:, t] = h
    return hs


def _lstm(W_i, b_i, W_h, b_h, xs):
    B, T, _ = xs.shape
    nh = W_h.shape[0]
    h = np.zeros((B, nh)); c = np.zeros((B, nh))
    hs = np.zeros((B, T, nh))
    for t in range(T):
        g = xs[:, t] @ W_i + b_i + h @ W_h + b_h
        i = _sigmoid(g[:, :nh])
        f = _sigmoid(g[:, nh:2 * nh])
        gg = np.tanh(g[:, 2 * nh:3 * nh])
        o = _sigmoid(g[:, 3 * nh:])
        c = f * c + i * gg
        h = o * np.tanh(c)
        hs[:, t] = h
    return hs


class RDP:
    """Exported residual dynamics predictor.

    ``predict(window)`` takes ``(B, H, 26)`` frames in the order of (6.2) and
    returns ``(B, 6)`` in **N and N.m**, after inverse normalisation.
    """

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.meta = json.loads(str(z["__meta__"]))
        self.W = {k: np.asarray(z[k], dtype=np.float64) for k in z.files
                  if k != "__meta__"}
        self.kind = self.meta["kind"]
        self.H = int(self.meta["H"])
        self.frame_dim = int(self.meta["frame_dim"])
        self.mu = np.asarray(self.meta["mu"], dtype=np.float64)
        self.sd = np.asarray(self.meta["sd"], dtype=np.float64)
        self.out_sd = np.asarray(self.meta["out_sd"], dtype=np.float64)
        self.channels = list(self.meta["channels"])
        self.units = list(self.meta["units"])

    def predict(self, win):
        win = np.asarray(win, dtype=np.float64)
        if win.ndim == 2:
            win = win[None]
        if win.shape[1] != self.H or win.shape[2] != self.frame_dim:
            raise ValueError(f"window must be (B,{self.H},{self.frame_dim}), "
                             f"got {win.shape}")
        x = (win - self.mu) / self.sd
        if self.kind == "GRU":
            h = _gru(self.W["l1.Wi"], self.W["l1.bi"], self.W["l1.Wh"],
                     self.W["l1.bh"], x)
            h = _gru(self.W["l2.Wi"], self.W["l2.bi"], self.W["l2.Wh"],
                     self.W["l2.bh"], h)
            feat = h[:, -1]
        elif self.kind == "LSTM":
            h = _lstm(self.W["l1.Wi"], self.W["l1.bi"], self.W["l1.Wh"],
                      self.W["l1.bh"], x)
            h = _lstm(self.W["l2.Wi"], self.W["l2.bi"], self.W["l2.Wh"],
                      self.W["l2.bh"], h)
            feat = h[:, -1]
        elif self.kind == "TCN":
            nb = int(self.meta["n_block"])
            for i in range(nb):
                d = 2 ** i
                hh = _relu(_causal_conv(self.W[f"b{i}.c1.W"], self.W[f"b{i}.c1.b"], x, d))
                hh = _relu(_causal_conv(self.W[f"b{i}.c2.W"], self.W[f"b{i}.c2.b"], hh, d))
                # the 1x1 residual projection is always exported, so there is
                # no branch here to fall out of step with the JAX side
                x = hh + _causal_conv(self.W[f"b{i}.res.W"],
                                      self.W[f"b{i}.res.b"], x, 1)
            feat = x[:, -1]
        elif self.kind == "CNN":
            nl = int(self.meta["n_layer"])
            for i in range(nl):
                x = _relu(_causal_conv(self.W[f"c{i}.W"], self.W[f"c{i}.b"], x, 1))
                if i < nl - 1 and x.shape[1] >= 4:
                    # anchored to the RIGHT edge: keeps the most recent frame
                    x = x[:, (x.shape[1] - 1) % 2::2]
            feat = x[:, -1]
        else:
            raise ValueError(self.kind)
        return (feat @ self.W["head.W"] + self.W["head.b"]) * self.out_sd


def load(npz_path):
    return RDP(npz_path)
