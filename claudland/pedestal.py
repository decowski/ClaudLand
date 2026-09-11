"""Pedestal handling for ATWD waveforms.

Two pedestal sources are combined:

1. the ``CmpPedestal`` table stored in the first event of every ``.sfz`` file
   (the table used by the compressor -- indexed by front-end channel, not by
   cable), and
2. the *pedestal-trigger* events (``PedestalA``/``PedestalB``, 50 of each at
   the start of every run) which digitise all three gains of every channel
   with no light in the detector.  Their average, with the quality cuts of
   ``Kat/src/KPedestalFifoManager.cc``, is the preferred pedestal.

After pedestal subtraction each waveform still shows a slowly varying
baseline offset; :func:`baseline` estimates it from the flat part of the
waveform (port of ``KPedestalFifoManager::baseline_shift``).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .wfcomp import NSAMPLES, ConnectionTable, CompressionPedestals, Waveform, WaveformBatch

__all__ = ["PedestalManager", "baseline", "SATURATED"]

SATURATED = 1023
MAX_CABLE = 2144
N_GAINS_PED = 3


class PedestalManager:
    """Per (cable, ATWD, gain) average pedestal waveforms."""

    def __init__(self, max_cable: int = MAX_CABLE):
        self.max_cable = max_cable
        self._sum = np.zeros((max_cable, 2, N_GAINS_PED, NSAMPLES), dtype=np.float64)
        self._sum2 = np.zeros_like(self._sum)
        self._n = np.zeros((max_cable, 2, N_GAINS_PED), dtype=np.int32)
        self.connection: Optional[ConnectionTable] = None
        self.compression: Optional[CompressionPedestals] = None
        self.n_rejected = 0

    # -- fallback table -------------------------------------------------------
    def set_compression_pedestals(self, connection: ConnectionTable, pedestals: CompressionPedestals) -> None:
        """Register the ``CmpPedestal`` table of the file as fallback for channels without measured pedestals."""
        self.connection = connection
        self.compression = pedestals

    # -- accumulation ---------------------------------------------------------
    @staticmethod
    def is_good_pedestal(samples: np.ndarray) -> bool:
        """Quality cuts of ``KPedestalFifoManager::set``: flat and without steps."""
        s = samples.astype(np.int32)
        if s.max() - s.min() >= 50:
            return False
        d = np.diff(s)
        return bool(np.all(np.abs(d) <= 40))

    def add(self, wf: Waveform) -> bool:
        """Add one waveform from a pedestal-trigger event."""
        if wf.gain >= N_GAINS_PED or wf.cable >= self.max_cable or abs(wf.launch_offset) > 1:
            return False
        if not self.is_good_pedestal(wf.samples):
            self.n_rejected += 1
            return False
        s = wf.samples.astype(np.float64)
        self._sum[wf.cable, wf.atwd, wf.gain] += s
        self._sum2[wf.cable, wf.atwd, wf.gain] += s * s
        self._n[wf.cable, wf.atwd, wf.gain] += 1
        return True

    def add_event(self, waveforms: Iterable[Waveform]) -> int:
        """Add all waveforms of a pedestal-trigger event; returns the number accepted."""
        if isinstance(waveforms, WaveformBatch):
            return self.add_batch(waveforms)
        return sum(self.add(w) for w in waveforms)

    def add_batch(self, b: WaveformBatch) -> int:
        """Vectorised :meth:`add` for a whole :class:`WaveformBatch`."""
        if len(b) == 0:
            return 0
        s = b.samples.astype(np.int32)
        ok = (b.gain < N_GAINS_PED) & (b.cable < self.max_cable) & (np.abs(b.launch) <= 1)
        flat = (s.max(axis=1) - s.min(axis=1) < 50) & (np.abs(np.diff(s, axis=1)).max(axis=1) <= 40)
        self.n_rejected += int((ok & ~flat).sum())
        use = ok & flat
        if not use.any():
            return 0
        sf = s[use].astype(np.float64)
        c, a, g = b.cable[use], b.atwd[use], b.gain[use]
        np.add.at(self._sum, (c, a, g), sf)
        np.add.at(self._sum2, (c, a, g), sf * sf)
        np.add.at(self._n, (c, a, g), 1)
        return int(use.sum())

    # -- lookup ---------------------------------------------------------------
    def count(self, cable: int, atwd: int, gain: int) -> int:
        """Number of pedestal waveforms accumulated for the channel."""
        return int(self._n[cable, atwd, gain]) if gain < N_GAINS_PED else 0

    def has_measured(self, cable: int, atwd: int, gain: int) -> bool:
        """True if at least one pedestal waveform was accumulated for the channel."""
        return self.count(cable, atwd, gain) > 0

    def get(self, cable: int, atwd: int, gain: int) -> np.ndarray:
        """Average pedestal (float64[128]); falls back to the compression table, then zeros."""
        if gain < N_GAINS_PED and cable < self.max_cable and self._n[cable, atwd, gain] > 0:
            return self._sum[cable, atwd, gain] / self._n[cable, atwd, gain]
        if self.connection is not None and self.compression is not None and cable < len(self.connection):
            row = self.connection.pedestal_row(cable, atwd, gain)
            if row >= 0:
                return self.compression[row].astype(np.float64)
        return np.zeros(NSAMPLES)

    def get_batch(self, cable: np.ndarray, atwd: np.ndarray, gain: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`get`: ``float64[n, 128]`` pedestals for arrays of channels."""
        cable = np.asarray(cable, dtype=np.int64); atwd = np.asarray(atwd, dtype=np.int64)
        gain = np.asarray(gain, dtype=np.int64)
        out = np.zeros((len(cable), NSAMPLES))
        g = np.minimum(gain, N_GAINS_PED - 1); c = np.minimum(cable, self.max_cable - 1)
        n = self._n[c, atwd, g]
        meas = (gain < N_GAINS_PED) & (cable < self.max_cable) & (n > 0)
        if meas.any():
            out[meas] = self._sum[c[meas], atwd[meas], g[meas]] / n[meas][:, None]
        rest = ~meas
        if rest.any() and self.connection is not None and self.compression is not None:
            cc = cable[rest]
            inrange = cc < len(self.connection)
            base = np.where(inrange, self.connection.base_row[np.minimum(cc, len(self.connection) - 1)], -1)
            row = base + atwd[rest] * 4 + gain[rest]
            good = base >= 0
            idx = np.flatnonzero(rest)[good]
            out[idx] = self.compression.table[row[good]].astype(np.float64)
        return out

    def rms(self, cable: int, atwd: int, gain: int) -> np.ndarray:
        """Sample-by-sample rms of the accumulated pedestal waveforms (NaN with fewer than two)."""
        n = self.count(cable, atwd, gain)
        if n < 2:
            return np.full(NSAMPLES, np.nan)
        m = self._sum[cable, atwd, gain] / n
        v = self._sum2[cable, atwd, gain] / n - m * m
        return np.sqrt(np.clip(v, 0, None))

    def subtract(self, wf: Waveform) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(pedestal-subtracted float waveform, saturated-sample mask)``."""
        s = wf.samples.astype(np.float64)
        sat = wf.samples >= SATURATED
        return s - self.get(wf.cable, wf.atwd, wf.gain), sat

    def summary(self) -> str:
        """One-line summary of the pedestal coverage."""
        n_id = int((self._n[:1879, :, 0] > 0).sum())
        return (f"PedestalManager: {n_id}/{2 * 1879} ID (cable,ATWD) high-gain pedestals measured, "
                f"mean samples/channel {self._n[self._n > 0].mean() if (self._n > 0).any() else 0:.1f}, "
                f"{self.n_rejected} waveforms rejected")


def baseline(wave: np.ndarray, sat: Optional[np.ndarray] = None) -> float:
    """Estimate the flat baseline of a pedestal-subtracted waveform (ADC counts).

    Port of ``KPedestalFifoManager::baseline_shift``: samples on a slope or on a
    "table top" (pulse) are masked using a 7-point running slope and a level
    cut at 5% of the waveform range above the running-average minimum; the
    mean of the remaining samples is the baseline.
    """
    w = np.asarray(wave, dtype=np.float64)
    if sat is not None and sat.any():
        w = w.copy()
        w[sat] = np.nan
    # 7-point running slope
    d = np.empty(127)
    d[0] = w[1] - w[0]
    d[1] = (w[3] - w[0]) / 3.0
    d[2] = (w[5] - w[0]) / 5.0
    d[3:124] = (w[7:128] - w[0:121]) / 7.0
    d[124] = (w[127] - w[122]) / 5.0
    d[125] = (w[127] - w[124]) / 3.0
    d[126] = w[127] - w[126]
    # 3-point running average min/max
    ra = (w[:-2] + w[1:-1] + w[2:]) / 3.0
    finite = np.isfinite(ra)
    if not finite.any():
        return 0.0
    minval = np.nanmin(ra)
    maxval = np.nanmax(ra)
    rng = maxval - minval
    ulimit = max(rng * 0.05, 10.0) + minval
    slope_cut = np.where(np.arange(127) < 10, rng * 0.01, rng * 0.001)
    good = np.ones(128, dtype=bool)
    good[:127] &= np.abs(d) <= slope_cut
    good &= w <= ulimit
    good &= np.isfinite(w)
    if good.sum() > 0:
        return float(w[good].mean())
    total = np.nansum(w)
    if total < 0:
        return float(total / 128.0)
    return float(minval) if minval > 0 else 0.0


def baseline_batch(waves: np.ndarray) -> np.ndarray:
    """Vectorised version of :func:`baseline` for an ``(n, 128)`` array."""
    w = np.asarray(waves, dtype=np.float64)
    n = w.shape[0]
    d = np.empty((n, 127))
    d[:, 0] = w[:, 1] - w[:, 0]
    d[:, 1] = (w[:, 3] - w[:, 0]) / 3.0
    d[:, 2] = (w[:, 5] - w[:, 0]) / 5.0
    d[:, 3:124] = (w[:, 7:128] - w[:, 0:121]) / 7.0
    d[:, 124] = (w[:, 127] - w[:, 122]) / 5.0
    d[:, 125] = (w[:, 127] - w[:, 124]) / 3.0
    d[:, 126] = w[:, 127] - w[:, 126]
    ra = (w[:, :-2] + w[:, 1:-1] + w[:, 2:]) / 3.0
    minval = np.nanmin(ra, axis=1)
    maxval = np.nanmax(ra, axis=1)
    rng = maxval - minval
    ulimit = np.maximum(rng * 0.05, 10.0) + minval
    slope_cut = np.where(np.arange(127) < 10, rng[:, None] * 0.01, rng[:, None] * 0.001)
    good = np.ones_like(w, dtype=bool)
    good[:, :127] &= np.abs(d) <= slope_cut
    good &= w <= ulimit[:, None]
    good &= np.isfinite(w)
    ngood = good.sum(axis=1)
    s = np.where(good, w, 0.0).sum(axis=1)
    out = np.where(ngood > 0, s / np.maximum(ngood, 1), 0.0)
    fallback = ngood == 0
    if fallback.any():
        total = np.nansum(w[fallback], axis=1)
        alt = np.where(total < 0, total / 128.0, np.where(minval[fallback] > 0, minval[fallback], 0.0))
        out[fallback] = alt
    return out
