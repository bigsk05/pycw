"""CW decode rules: Otsu threshold, morphology, run segmentation, scored
dit estimator, Morse mapping. Pure numpy; shared logic with gocw/cw.
"""
import numpy as np

from ..morse import DIT, DAH, MORSE_TABLE

# code string ('.'/' -') -> char, built from the canonical table in pycw.morse
REV = {
    "".join("." if s is DIT else "-" for s in symbols): ch
    for ch, symbols in MORSE_TABLE.items()
}


# ---------------- segmentation helpers ----------------
def runs_from_binary(b):
    """All runs (marks AND gaps) with values, starting from state 0."""
    b = b.astype(np.int8)
    d = np.diff(np.concatenate(([0], b, [0])))
    trans = np.flatnonzero(d != 0)          # state-change positions
    runs = []
    prev = 0
    curv = 0
    for i in trans:
        if i > prev:
            runs.append((prev, i, curv))
        curv = b[i] if i < len(b) else curv
        prev = i
    if prev < len(b):
        runs.append((prev, len(b), curv))
    return runs


def _morph(b, k, mode):
    if k <= 1 or len(b) <= 2 * k:
        return b
    kernel = np.ones(k)
    c = np.convolve(b.astype(np.float32), kernel, mode='same')
    if mode == 'open':        # remove mark spikes shorter than k, then dilate
        eroded = c >= k
        c2 = np.convolve(eroded.astype(np.float32), kernel, mode='same')
        return c2 >= 1
    else:                     # close: fill gaps shorter than k, then erode
        dilated = c >= 1
        c2 = np.convolve(dilated.astype(np.float32), kernel, mode='same')
        return c2 >= k


def otsu_threshold(p, bins=64):
    """Per-file adaptive decision threshold (Otsu between the two clusters)."""
    p = np.asarray(p, np.float32)
    if p.min() >= p.max():
        return 0.5
    hist, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.5
    w0 = np.cumsum(hist); w1 = total - w0
    m0 = np.cumsum(hist * centers) / (w0 + 1e-9)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / (w1 + 1e-9)
    between = w0 * w1 * (m0 - m1) ** 2
    thr = centers[int(np.argmax(between))]
    return float(np.clip(thr, 0.1, 0.9))


def estimate_dit(mark_lens, gap_lens=None, lo=3.0, hi=60.0):
    """Robust dit estimate (frames) via scored candidate search.

    Pools mark (1x/3x) and gap (1x/3x/7x) durations and picks the candidate
    with the highest fraction of explained runs; ties favour the smallest
    candidate (avoids 3x aliasing). lo/hi frame a realistic WPM range and
    resist tiny noise fragments (QRM/impulses on long files) skewing the log-
    gap heuristic.
    """
    marks = np.array([m for m in mark_lens if lo <= m], np.float32)
    gaps = np.array(gap_lens, np.float32) if gap_lens is not None else \
        np.array([], np.float32)
    if len(marks) < 3:
        return 10.0 if len(marks) == 0 else float(np.median(marks))

    def score(arr, mults):
        if len(arr) == 0:
            return 0.0
        rel = arr[:, None] / dit / mults[None, :]    # express in dit units
        return (np.abs(rel - 1) <= 0.35).any(axis=1).mean()

    cands = np.unique(np.clip(marks, lo, hi))
    best_score, best = -1.0, 10.0
    for dit in cands:
        sc = score(marks, np.array([1.0, 3.0]))
        if len(gaps):
            sc += 0.7 * score(gaps, np.array([1.0, 3.0, 7.0]))
        if sc > best_score * 0.99:
            if sc > best_score or dit < best:
                best_score, best = sc, float(dit)
    return max(best, 2.5)


def decode_runs(probs, thr=None, smooth=2, morph=2):
    """Per-frame P(mark) -> text via run timing + Morse table.

    thr=None uses a per-file Otsu threshold. morph (frames @200Hz) removes
    spurious dits/gaps; decode also drops mark runs below ~0.55 dit (impulse
    noise) so they never inflate into spurious symbols.
    """
    p = np.asarray(probs, np.float32)
    thr = otsu_threshold(p) if thr is None else thr
    if smooth > 1:
        k = np.ones(smooth) / smooth
        p = np.convolve(p, k, mode='same')
    on = p >= thr
    on = _morph(on, morph, 'open')
    on = _morph(on, morph, 'close')
    runs = runs_from_binary(on)
    marks = [e - s for s, e, v in runs if v == 1]
    gaps = [e - s for s, e, v in runs if v == 0]
    dit = estimate_dit(marks, gaps)
    min_mark = max(3.0, 0.55 * dit)
    code = ''
    out = []
    conf = 1.0
    started = False
    for idx, (s, e, v) in enumerate(runs):
        if v == 1:
            L = e - s
            if L < min_mark:
                continue                       # impulse / fragment
            sym = '.' if L < 2.0 * dit else '-'
            code += sym
            conf = min(conf, float(p[s:e].mean()))
        else:
            g = e - s
            if not started:
                continue
            if g < 2.2 * dit:
                continue
            if code:
                out.append((REV.get(code, '?'), conf))
            if g >= 4.6 * dit and out:      # word space ~6 dit (jitter-tolerant)
                out.append((' ', conf))
            code = ''
            conf = 1.0
        started = True
    if code:
        out.append((REV.get(code, '?'), conf))
    return ''.join(ch for ch, _ in out), dit


