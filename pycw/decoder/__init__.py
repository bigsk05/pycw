"""pycw.decoder — numpy-only CW (Morse) audio decoder.

Runtime dependency: numpy only. No PyTorch is imported anywhere on this path;
the cross-runtime weights (``model.bin``, exported by the ``cw-train``
repository) ship inside this package.

Pipeline mirrors the Go decoder in gocw/cw exactly (same weight file, same
float64 math), so both implementations reproduce the golden vectors
(golden.bin in the cw-train repo) bit-for-bit.
"""

from .np_model import NPModel, load_weights
from .rules import decode_runs
from .features import extract_features, detect_tone

__all__ = ["Decoder", "decode_wav", "decode_samples", "decode_bytes",
           "NPModel", "load_weights", "detect_tone"]

import os

import numpy as np


def model_bytes():
    """The embedded cross-runtime weight file (model.bin)."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "model.bin"), "rb") as f:
        return f.read()


class Decoder:
    """Small object wrapping the numpy inference + decode rules."""

    def __init__(self, model=None, tone=None):
        if model is None:
            model = NPModel(load_weights(model_bytes()))
        self.model = model
        self.tone = tone  # optional fixed tone override; else auto-detected

    def decode(self, samples, sample_rate=16000, tone=None):
        """samples: float32 mono in [-1, 1] (any sample rate). Returns text."""
        f0 = detect_tone(samples, sample_rate) if tone is None else tone
        F, _ = extract_features(samples, sample_rate, f0=f0)
        p = self.model.forward(F)
        return decode_runs(p)[0]


def decode_samples(samples, sample_rate=16000, tone=None, decoder=None):
    """Decode raw PCM samples (float32 mono in [-1,1]) to Morse text."""
    if decoder is None:
        decoder = Decoder(tone=tone)
    return decoder.decode(samples, sample_rate, tone=tone)


def _read_wav(path):
    """Read a WAV file -> (float32 mono samples in [-1,1], sample rate)."""
    import wave

    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        a = (a - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        a = (b[:, 0].astype(np.int32) |
             (b[:, 1].astype(np.int32) << 8) |
             (b[:, 2].astype(np.int32) << 16))
        a = (a - (1 << 23)).astype(np.float32) / float(1 << 23)
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a, sr


def decode_wav(path, tone=None, decoder=None):
    """Decode a WAV file to Morse text (auto-detects tone, any sample rate)."""
    if decoder is None:
        decoder = Decoder(tone=tone)
    samples, sr = _read_wav(path)
    return decoder.decode(samples, sr, tone=tone)


def decode_bytes(pcm, sample_rate=16000, channels=1, tone=None, decoder=None):
    """Decode raw little-endian int16 PCM bytes (like gocw.Generate output)."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        a = a.reshape(-1, channels).mean(axis=1)
    return decode_samples(a, sample_rate, tone=tone, decoder=decoder)