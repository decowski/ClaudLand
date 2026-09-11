"""Time and charge ("TQ") extraction from ATWD waveforms.

Two pulse finders are provided:

* :func:`multi_tq` -- a faithful port of Kunio Inoue's ``KMultiTQ::get_multi_tq``
  (``Kat/src/KMultiTQ.cc``): the waveform is smoothed through a 5-point
  running-average first derivative, peaks are found from the derivatives,
  leading/trailing edges are located around each peak and the charge is the
  area between the edges.  Handles multiple pulses per waveform.
* :func:`fast_tq` -- a vectorised single-pulse version operating on an
  ``(n, 128)`` array at once (same smoothing, same edge definitions, but only
  the highest pulse is characterised).  About two orders of magnitude faster.

Units: times are in *samples* (multiply by the ATWD sampling period, ~1.49 ns,
and subtract ``launch_offset * 25 ns``); charges are in ADC-count x samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = ["Pulse", "smooth", "smooth_batch", "multi_tq", "fast_tq", "MIN_PEAK_HEIGHT"]

MIN_PEAK_HEIGHT = 5.0   # ADC counts above the previous minimum (KMultiTQ::find_peaks)
MAX_PEAKS = 15


@dataclass
class Pulse:
    """One pulse found in a waveform: leading/CFD/peak times (samples), height and charge (ADC)."""
    peak: float      #: sample index of the peak
    lead: float      #: leading edge (sample)
    trail: float     #: trailing edge (sample)
    charge: float    #: area between lead and trail (ADC x sample)
    height: float    #: smoothed peak height (ADC)


def _derivatives(wave: np.ndarray):
    """5/3/3-point running-average derivatives and the smoothed waveform."""
    w = wave
    dif1 = np.empty(127)
    dif1[0] = w[1] - w[0]
    dif1[1] = (w[3] - w[0]) / 3.0
    dif1[2:125] = (w[5:128] - w[0:123]) / 5.0
    dif1[125] = (w[127] - w[124]) / 3.0
    dif1[126] = w[127] - w[126]
    dif2 = np.empty(126)
    dif2[0] = dif1[1] - dif1[0]
    dif2[1:125] = (dif1[3:127] - dif1[0:124]) / 3.0
    dif2[125] = dif1[126] - dif1[125]
    dif3 = np.empty(125)
    dif3[0] = dif2[1] - dif2[0]
    dif3[1:124] = (dif2[3:126] - dif2[0:123]) / 3.0
    dif3[124] = dif2[125] - dif2[124]
    wf = np.empty(128)
    wf[0] = w[0]
    wf[1:] = w[0] + np.cumsum(dif1)
    wf += (w.sum() - wf.sum()) / 128.0   # baseline adjustment
    return wf, dif1, dif2, dif3


def smooth(wave: np.ndarray) -> np.ndarray:
    """The smoothed waveform used by the TQ algorithms."""
    return _derivatives(np.asarray(wave, dtype=np.float64))[0]


def _find_peaks(wave, wf, dif1, dif2, dif3) -> List[int]:
    """Port of ``KMultiTQ::find_peaks``."""
    peaks: List[int] = []
    current_max = wf[0]
    previous_min = 0.0
    previous_height = 0.0
    previous_peak = -1000
    over_flow = 0
    for i in range(4, 125):
        if (wf[i] - previous_min >= MIN_PEAK_HEIGHT) and wf[i] > current_max * 0.1 and wave[i] < 1023.0:
            if not ((i - previous_peak) < 30 and wf[i] < previous_height * 0.2):
                if ((dif1[i - 1] * dif1[i + 1] < 0.0 and dif2[i] <= 0.0)
                        or (dif1[i - 1] * dif1[i + 1] == 0.0 and dif2[i] <= 0.0 and wf[i] > current_max)
                        or (dif2[i - 1] * dif2[i + 1] < 0.0 and dif1[i] > 0.0 and dif3[i] > 0.0)):
                    skip = False
                    if (i - previous_peak) < 4:
                        if current_max >= wf[i]:
                            skip = True
                        else:
                            peaks.pop()
                    if not skip:
                        peaks.append(i)
                        previous_height = wf[i]
                        previous_peak = i
                        previous_min = 0.0
        if wave[i] == 1023:
            if over_flow == 0:
                peaks.append(i)
                previous_height = wf[i]
                previous_peak = i
                previous_min = 0.0
            over_flow += 1
        if wf[i] > current_max:
            current_max = wf[i]
        if wf[i] < previous_min:
            previous_min = wf[i]
        if len(peaks) == MAX_PEAKS:
            break
    return peaks


def multi_tq(wave: np.ndarray) -> Tuple[List[Pulse], float, np.ndarray]:
    """Find all pulses in one baseline-subtracted waveform.

    Returns ``(pulses, total_charge, smoothed_waveform)``.  Port of
    ``KMultiTQ::get_multi_tq``.
    """
    wave = np.asarray(wave, dtype=np.float64)
    wf, dif1, dif2, dif3 = _derivatives(wave)
    peaks = _find_peaks(wave, wf, dif1, dif2, dif3)
    npeak = len(peaks)
    if npeak == 0:
        return [], 0.0, wf
    ts = [0.0] * (npeak + 1)
    te = [0.0] * npeak
    # leading edge of the first pulse
    peak = peaks[0]
    start = peak - 10
    ok = 0
    for i in range(peak - 1, -1, -1):
        if wf[i] < 0.0:
            start = i
            break
        if dif2[i] < 0.0:
            ok = 1
            continue
        if ok == 0:
            continue
        if wf[peak] - wf[i] <= 5.0:
            continue
        if wf[i] - wf[0] > wf[peak] * 0.05 + 5.0:
            continue
        if dif1[i] <= 0.0:
            start = i
            break
        if dif3[i] > 0.0:
            start = i
            break
    ts[0] = float(start)
    # leading edges of subsequent pulses
    for j in range(1, npeak):
        start = int(peaks[j] + peaks[j - 1]) // 2
        ok = 0
        for i in range(int(peaks[j]) - 1, int(peaks[j - 1]), -1):
            if wf[i] < 0.0:
                start = i
                break
            if dif2[i] < 0.0:
                ok = 1
                continue
            if ok == 0:
                continue
            if dif1[i] < 1.0 and wf[i] < 2.0:
                start = i
                break
            if dif1[i] < 0.0:
                start = i
                break
            if dif3[i] > 0.0:
                start = i
                break
        ts[j] = float(start)
    # trailing edges
    ts[npeak] = 128.0
    for j in range(npeak):
        limit = int(wf[peaks[j]] * 1.5)
        if limit < 25:
            limit = 25
        limit += peaks[j]
        te[j] = ts[j + 1]
        if int(te[j]) > limit:
            te[j] = float(limit)
        else:
            limit = int(te[j])
        for i in range(peaks[j], limit):
            if wf[i] <= 0.0:
                te[j] = float(i)
                break
    # areas
    pulses: List[Pulse] = []
    total = 0.0
    for j in range(npeak):
        a = max(int(ts[j]), 0)
        b = int(te[j])
        q = float(wf[a:b].sum()) if b > a else 0.0
        total += q
        pulses.append(Pulse(float(peaks[j]), ts[j], te[j], q, float(wf[peaks[j]])))
    return pulses, total, wf


# ---------------------------------------------------------------------------
# vectorised pulse finder
# ---------------------------------------------------------------------------
def smooth_batch(waves: np.ndarray):
    """Smoothed waveforms and 5-point first derivatives for an ``(n, 128)`` array."""
    w = np.asarray(waves, dtype=np.float64)
    n = w.shape[0]
    dif1 = np.empty((n, 127))
    dif1[:, 0] = w[:, 1] - w[:, 0]
    dif1[:, 1] = (w[:, 3] - w[:, 0]) / 3.0
    dif1[:, 2:125] = (w[:, 5:128] - w[:, 0:123]) / 5.0
    dif1[:, 125] = (w[:, 127] - w[:, 124]) / 3.0
    dif1[:, 126] = w[:, 127] - w[:, 126]
    wf = np.empty((n, 128))
    wf[:, 0] = w[:, 0]
    wf[:, 1:] = w[:, :1] + np.cumsum(dif1, axis=1)
    wf += ((w.sum(axis=1) - wf.sum(axis=1)) / 128.0)[:, None]
    return wf, dif1


def fast_tq(waves: np.ndarray, sat: np.ndarray | None = None, min_height: float = MIN_PEAK_HEIGHT,
            cfd_fraction: float = 0.5, noise_fraction: float = 0.15) -> dict:
    """Vectorised TQ for an ``(n, 128)`` array of baseline-subtracted waveforms.

    Pulses are the contiguous regions where the smoothed waveform is > 0
    (Detwiler thesis Sect. 4.2.2).  A region is a *pulse* if its maximum is at
    least ``min_height`` ADC counts and its area is at least ``noise_fraction``
    of the summed area of all such regions.

    Returns a dict of per-waveform arrays:

    ``q_total``    summed area of all accepted pulses (ADC x sample)
    ``q_first``    area of the first accepted pulse
    ``npulse``     number of accepted pulses
    ``t_lead``     start (zero crossing) of the first pulse, in samples
    ``t_cfd``      time at which the first pulse crosses ``cfd_fraction`` of
                   its height (linear interpolation), in samples
    ``t_peak``     parabolic-interpolated position of the first pulse maximum
    ``height``     height of the first pulse; ``height_max`` largest pulse
    ``found``      at least one accepted pulse
    ``saturated``  any sample at the 10-bit ceiling
    ``smoothed``   the smoothed waveforms
    """
    w = np.asarray(waves, dtype=np.float64)
    n = w.shape[0]
    wf, dif1 = smooth_batch(w)
    if sat is None:
        sat = w >= 1023
    idx = np.arange(128)
    pos = wf > 0.0
    # --- label contiguous positive regions (global segment ids) ---------------
    flat_pos = pos.ravel()
    starts = flat_pos & ~np.concatenate(([False], flat_pos[:-1]))
    # force a new segment at each row boundary
    starts[::128] |= flat_pos[::128]
    seg_id = np.cumsum(starts) - 1          # -1 where not positive (fixed below)
    valid = flat_pos & (seg_id >= 0)
    nseg = max(int(starts.sum()), 1)
    flat_wf = wf.ravel()
    seg_area = np.bincount(seg_id[valid], weights=flat_wf[valid], minlength=nseg)
    seg_max = np.full(nseg, -np.inf)
    np.maximum.at(seg_max, seg_id[valid], flat_wf[valid])
    seg_row = np.bincount(seg_id[valid], weights=np.repeat(np.arange(n), 128)[valid], minlength=nseg)
    seg_cnt = np.bincount(seg_id[valid], minlength=nseg)
    seg_row = (seg_row / np.maximum(seg_cnt, 1)).astype(int)
    seg_start = np.full(nseg, 128)
    np.minimum.at(seg_start, seg_id[valid], (np.tile(idx, n))[valid])
    # --- accept pulses ----------------------------------------------------------
    keep = seg_max >= min_height
    row_total = np.bincount(seg_row[keep], weights=seg_area[keep], minlength=n)
    # noise rejection relative to the row total (one pass, like Detwiler's 15% cut)
    keep &= seg_area >= noise_fraction * row_total[seg_row]
    row_total = np.bincount(seg_row[keep], weights=seg_area[keep], minlength=n)
    npulse = np.bincount(seg_row[keep], minlength=n)
    # first accepted pulse per row
    first_seg = np.full(n, -1)
    order = np.lexsort((seg_start, seg_row))          # by row then start
    kept_sorted = order[keep[order]]
    rows_sorted = seg_row[kept_sorted]
    if len(kept_sorted):
        firstmask = np.concatenate(([True], rows_sorted[1:] != rows_sorted[:-1]))
        first_seg[rows_sorted[firstmask]] = kept_sorted[firstmask]
    found = first_seg >= 0
    fs = np.where(found, first_seg, 0)
    start = np.where(found, seg_start[fs], 0)
    height = np.where(found, seg_max[fs], 0.0)
    q_first = np.where(found, seg_area[fs], 0.0)
    height_max = np.full(n, 0.0)
    np.maximum.at(height_max, seg_row[keep], seg_max[keep])
    # --- timing of the first pulse -----------------------------------------------
    rows = np.arange(n)
    in_first = (np.tile(idx, n).reshape(n, 128) >= start[:, None]) & (seg_id.reshape(n, 128) == fs[:, None]) & pos
    peak = np.where(in_first, wf, -np.inf).argmax(axis=1)
    pm = np.clip(peak - 1, 0, 127); pp = np.clip(peak + 1, 0, 127)
    y0 = wf[rows, pm]; y1 = wf[rows, peak]; y2 = wf[rows, pp]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    t_peak = peak + np.clip(delta, -1, 1)
    thr = cfd_fraction * height
    rise = (idx[None, :] >= start[:, None]) & (idx[None, :] <= peak[:, None]) & (wf >= thr[:, None])
    k = np.where(rise.any(axis=1), rise.argmax(axis=1), peak)
    km = np.clip(k - 1, 0, 127)
    ya = wf[rows, km]; yb = wf[rows, k]
    frac = np.where((yb - ya) > 1e-9, (thr - ya) / np.where((yb - ya) == 0, 1, (yb - ya)), 0.0)
    t_cfd = np.where(k > start, km + np.clip(frac, 0, 1), k.astype(float))
    t_cfd = np.where(found, t_cfd, np.nan)
    t_peak = np.where(found, t_peak, np.nan)
    t_lead = np.where(found, start.astype(float), np.nan)
    return {
        "q_total": row_total, "q_first": q_first, "npulse": npulse,
        "t_lead": t_lead, "t_cfd": t_cfd, "t_peak": t_peak,
        "height": height, "height_max": height_max,
        "found": found, "saturated": sat.any(axis=1), "smoothed": wf,
    }
