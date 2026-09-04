"""Front-end DSP: down-convert -> I/Q + envelope -> AGC -> multiscale features.

Frame rate = 200 Hz. Channels (C=6):
    I, Q   : complex baseband (real/imag) after down-conversion + decimation
    env    : |baseband| rectified, normalized
    logenv : dB-compressed envelope
    m3, m9 : moving averages of log-envelope (15/45 ms) -> coarse matched bank
"""
import numpy as np

FEAT_RATE = 200


def detect_tone(x, sr, lo=350, hi=1700):
    """Robustly find the dominant CW tone by peak of magnitude spectrum."""
    seg = x[: int(1.5 * sr)]
    if len(seg) < 1024:
        seg = x
    w = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * w))
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)
    m = (f >= lo) & (f <= hi)
    if m.sum() == 0:
        return 700.0
    i = int(np.argmax(S[m]))
    return float(f[m][i])


def envelope_iq(x, sr, f0):
    """Complex baseband -> decimated I, Q, |env| at FEAT_RATE."""
    n = len(x)
    t = np.arange(n) / sr
    y = x * np.exp(-2j * np.pi * f0 * t)   # complex baseband
    block = max(1, int(round(sr / FEAT_RATE)))
    k = n // block
    y = y[: k * block].reshape(k, block)
    I = y.real.mean(axis=1)
    Q = y.imag.mean(axis=1)
    env = np.abs(y).mean(axis=1)
    return I.astype(np.float64), Q.astype(np.float64), env.astype(np.float64)


def extract_features(x, sr=16000, f0=None, gain=1.0):
    """Full front-end. Returns (F, 6) float32 and detected f0.

    gain simulates recording/AGC gain variation (0.3-3x) to force scale
    invariance. All channels are scaled by the same per-file robust peak
    denominator so input stays bounded and gain-invariant.
    """
    f0 = detect_tone(x, sr) if f0 is None else f0
    I, Q, env = envelope_iq(x * gain, sr, f0)
    e = np.log1p(np.maximum(env, 0.0))
    med = np.median(e)
    hi = np.quantile(e, 0.985)
    denom = (hi - med) + 1e-9
    if denom < 1e-6:
        denom = (np.std(e) + 1e-9)
    logenv = (e - med) / denom
    pe = np.quantile(env, 0.985) + 1e-9
    envn = env / pe
    pI = np.quantile(np.abs(I), 0.985) + 1e-9
    pQ = np.quantile(np.abs(Q), 0.985) + 1e-9
    In = I / pI
    Qn = Q / pQ
    m3 = np.convolve(logenv, np.ones(3) / 3, mode='same')
    m9 = np.convolve(logenv, np.ones(9) / 9, mode='same')
    F = np.stack([In, Qn, envn, logenv, m3, m9], axis=-1)
    return F.astype(np.float32), f0


def labels_from_on(on, sr=16000):
    """Ground-truth mark/silence labels at FEAT_RATE from sample-res keying."""
    block = max(1, int(round(sr / FEAT_RATE)))
    k = len(on) // block
    lab = on[: k * block].reshape(k, block).mean(axis=1) > 0.5
    return lab.astype(np.float32)
