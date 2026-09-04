"""NumPy-only model forward: reads the cross-runtime ``model.bin`` weights
(exported by cw-train) and runs the exact same float64 math as the Go decoder
in gocw/cw and the PyTorch reference. No torch import.
"""
import struct

import numpy as np

BN_EPS = 1e-5


def load_weights(data):
    """Parse model.bin bytes (format spec in cw-train README)."""
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:  # path
        raw = open(data, 'rb').read()
    assert raw[:4] == b'CWM1', 'bad model.bin magic'
    n = struct.unpack('<q', raw[4:12])[0]
    pos = 12
    ts = {}
    for _ in range(n):
        ns = struct.unpack('<i', raw[pos:pos + 4])[0]
        pos += 4
        name = raw[pos:pos + ns].decode()
        pos += ns
        elems = struct.unpack('<q', raw[pos:pos + 8])[0]
        pos += 8
        data = np.frombuffer(raw[pos:pos + 4 * elems], dtype=np.float32)
        pos += 4 * elems
        ts[name] = data.astype(np.float64)  # weights kept in float64
    return ts


class NPModel:
    """NumPy forward pass (float64; identical math to gocw/cw and torch)."""

    def __init__(self, ts):
        self.ts = ts
        self.H = 64

    def _conv(self, x, w, b):
        T, _ = x.shape
        O, _, K = w.shape
        pad = K // 2
        xp = np.pad(x, ((pad, pad), (0, 0)))
        y = np.zeros((T, O), dtype=np.float64)
        for j in range(K):
            y += xp[j:j + T, :] @ w[:, :, j].T
        return y + b

    def _bn_relu(self, x, w, b, m, v):
        return np.maximum((x - m) / np.sqrt(v + BN_EPS) * w + b, 0)

    def _gru(self, x, ih, hh, bih, bhh):
        T, inC = x.shape
        H = self.H
        h = np.zeros(H, dtype=np.float64)
        out = np.empty((T, H), dtype=np.float64)
        ir, iz, inn = ih[0:H], ih[H:2 * H], ih[2 * H:3 * H]
        hr, hz, hn = hh[0:H], hh[H:2 * H], hh[2 * H:3 * H]
        bir, biz, bin_ = bih[0:H], bih[H:2 * H], bih[2 * H:3 * H]
        bhr, bhz, bhn = bhh[0:H], bhh[H:2 * H], bhh[2 * H:3 * H]
        for t in range(T):
            xt = x[t]
            r = 1 / (1 + np.exp(-(xt @ ir.T + h @ hr.T + bir + bhr)))
            z = 1 / (1 + np.exp(-(xt @ iz.T + h @ hz.T + biz + bhz)))
            hp = h @ hn.T + bhn
            n = np.tanh(xt @ inn.T + bin_ + r * hp)
            h = (1 - z) * n + z * h
            out[t] = h
        return out

    def forward(self, F):
        """F: (T, 6) float32/64 features -> p (T,) float64."""
        ts = self.ts
        H = self.H
        x = F.astype(np.float64)
        h1 = self._bn_relu(self._conv(x, ts['conv.0.weight'].reshape(32, 6, 9),
                                      ts['conv.0.bias']),
                           ts['conv.1.weight'], ts['conv.1.bias'],
                           ts['conv.1.running_mean'], ts['conv.1.running_var'])
        h2 = self._bn_relu(self._conv(h1, ts['conv.3.weight'].reshape(64, 32, 7),
                                      ts['conv.3.bias']),
                           ts['conv.4.weight'], ts['conv.4.bias'],
                           ts['conv.4.running_mean'], ts['conv.4.running_var'])
        h3 = self._bn_relu(self._conv(h2, ts['conv.6.weight'].reshape(64, 64, 5),
                                      ts['conv.6.bias']),
                           ts['conv.7.weight'], ts['conv.7.bias'],
                           ts['conv.7.running_mean'], ts['conv.7.running_var'])
        ih = ts['gru.weight_ih_l0'].reshape(3 * H, 64)
        hh = ts['gru.weight_hh_l0'].reshape(3 * H, 64)
        fwd = self._gru(h3, ih, hh,
                        ts['gru.bias_ih_l0'], ts['gru.bias_hh_l0'])
        rev = self._gru(h3[::-1], ts['gru.weight_ih_l0_reverse'].reshape(3 * H, 64),
                        ts['gru.weight_hh_l0_reverse'].reshape(3 * H, 64),
                        ts['gru.bias_ih_l0_reverse'], ts['gru.bias_hh_l0_reverse'])
        hi = np.concatenate([fwd, rev[::-1]], axis=1)   # (T, 128)
        yy = np.maximum(hi @ ts['head.0.weight'].reshape(64, 128).T +
                        ts['head.0.bias'], 0)
        logit = yy @ ts['head.2.weight'].reshape(-1) + ts['head.2.bias'][0]
        return 1 / (1 + np.exp(-logit))