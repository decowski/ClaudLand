"""Run-dependent calibration constants: ATWD sampling period, 1 p.e. charge,
per-channel time offsets, and gain ratios.

Defaults follow ``Kat/src/KTimeChargeCorrection.cc`` and
``Kat/src/KMultiTQ.cc``.  Values can be measured from the run itself with the
helpers in :mod:`claudland.reco` and stored as JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .wfcomp import NSAMPLES, Waveform

__all__ = ["TQCalibration", "ClockAnalyzer", "DEFAULT_BIN_NS", "DEFAULT_Q1PE",
           "LAUNCH_OFFSET_NS", "GAIN_FACTOR"]

MAX_CABLE = 2144
DEFAULT_BIN_NS = 1.49          # ATWD sampling period (KTimeChargeCorrection default)
DEFAULT_Q1PE = 200.0           # 1 p.e. ADC sum for the high-gain channel (Kat: 180..210)
LAUNCH_OFFSET_NS = 25.0        # one launch-offset unit = one 40 MHz clock tick
CLOCK_PERIOD_NS = 25.0
# high/medium/low amplifier gain ratios relative to high gain
GAIN_FACTOR = np.array([1.0, 4.86, 41.86, 1.0])


@dataclass
class TQCalibration:
    """Constants converting (sample, ADC sum) into (ns, p.e.)."""

    bin_ns: np.ndarray = field(default_factory=lambda: np.full((MAX_CABLE, 2), DEFAULT_BIN_NS))
    q1pe: np.ndarray = field(default_factory=lambda: np.full((MAX_CABLE, 2), DEFAULT_Q1PE))
    t0: np.ndarray = field(default_factory=lambda: np.zeros((MAX_CABLE, 2)))
    gain_factor: np.ndarray = field(default_factory=lambda: GAIN_FACTOR.copy())
    bad: np.ndarray = field(default_factory=lambda: np.zeros(MAX_CABLE, dtype=bool))
    meta: Dict[str, object] = field(default_factory=dict)
    eta: Optional[np.ndarray] = None    #: per-tube light yield at the centre (p.e./MeV), inner tubes
    dark: Optional[np.ndarray] = None   #: per-tube dark hits per event window, inner tubes
    eta_q: Optional[np.ndarray] = None  #: per-tube charge yield at the centre (p.e./MeV), for the charge estimator

    def time_ns(self, cable, atwd, lead_sample, launch_offset):
        """Hit time in ns relative to the trigger (vectorised)."""
        cable = np.asarray(cable); atwd = np.asarray(atwd)
        return (np.asarray(lead_sample) * self.bin_ns[cable, atwd]
                - np.asarray(launch_offset) * LAUNCH_OFFSET_NS
                - self.t0[cable, atwd])

    def charge_pe(self, cable, atwd, gain, adc_sum):
        """Convert ADC-sample sums to photoelectrons with the per-channel Q0 and the gain factor."""
        cable = np.asarray(cable); atwd = np.asarray(atwd)
        return np.asarray(adc_sum) / self.q1pe[cable, atwd] * self.gain_factor[np.asarray(gain)]

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        """Write the calibration (arrays and metadata) to a JSON file."""
        def _default(o):
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"cannot serialise {type(o).__name__}")
        with open(path, "w") as f:
            json.dump({
                "bin_ns": self.bin_ns.tolist(), "q1pe": self.q1pe.tolist(), "t0": self.t0.tolist(),
                "gain_factor": self.gain_factor.tolist(), "bad": self.bad.astype(int).tolist(),
                "meta": self.meta,
                "eta": None if self.eta is None else self.eta.tolist(),
                "dark": None if self.dark is None else self.dark.tolist(),
                "eta_q": None if self.eta_q is None else self.eta_q.tolist(),
            }, f, default=_default)

    @classmethod
    def load(cls, path: str) -> "TQCalibration":
        """Read a calibration written by :meth:`save`."""
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d["bin_ns"]), np.array(d["q1pe"]), np.array(d["t0"]),
                   np.array(d["gain_factor"]), np.array(d["bad"], dtype=bool), d.get("meta", {}),
                   None if d.get("eta") is None else np.array(d["eta"]),
                   None if d.get("dark") is None else np.array(d["dark"]),
                   None if d.get("eta_q") is None else np.array(d["eta_q"]))

    def summary(self) -> str:
        """One-line summary of the calibration contents."""
        b = self.bin_ns[:1879]
        return (f"bin width mean {b.mean():.4f} ns (rms {b.std():.4f}); q1pe mean {self.q1pe[:1879].mean():.1f}; "
                f"t0 rms {self.t0[:1879].std():.2f} ns; {self.bad.sum()} bad channels")


class ClockAnalyzer:
    """Measure the ATWD sampling period from the 40 MHz clock waveforms.

    Clock-trigger events (``ClockA``/``ClockB``, gain field 3) digitise the
    40 MHz trigger clock.  The period in samples is obtained from the FFT power
    spectrum of the pedestal-subtracted waveform with parabolic interpolation
    (Detwiler thesis Sect. 4.2.3); the sampling period is
    ``25 ns / period_samples``.  Results are averaged per (cable, ATWD).
    """

    def __init__(self, max_cable: int = MAX_CABLE, min_period: float = 12.0, max_period: float = 24.0):
        self._sum = np.zeros((max_cable, 2))
        self._n = np.zeros((max_cable, 2), dtype=np.int32)
        self.min_period = min_period
        self.max_period = max_period

    @staticmethod
    def period_samples(waves: np.ndarray, min_period=12.0, max_period=24.0) -> np.ndarray:
        """Clock period (samples) for each row of an ``(n, 128)`` array."""
        w = np.asarray(waves, dtype=np.float64)
        w = w - w.mean(axis=1, keepdims=True)
        # zero-pad to 1024 for finer frequency resolution
        spec = np.abs(np.fft.rfft(w, n=1024, axis=1)) ** 2
        freqs = np.fft.rfftfreq(1024)  # cycles per sample
        band = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
        s = np.where(band[None, :], spec, 0.0)
        k = s.argmax(axis=1)
        rows = np.arange(w.shape[0])
        y0 = spec[rows, np.clip(k - 1, 0, None)]; y1 = spec[rows, k]; y2 = spec[rows, np.clip(k + 1, 0, spec.shape[1] - 1)]
        den = y0 - 2 * y1 + y2
        delta = np.where(np.abs(den) > 0, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
        f = (k + np.clip(delta, -0.5, 0.5)) / 1024.0
        with np.errstate(divide="ignore"):
            period = np.where(f > 0, 1.0 / f, np.nan)
        # reject waveforms without a clear clock signal
        amp = w.std(axis=1)
        period[amp < 5.0] = np.nan
        return period

    def add_event(self, waveforms) -> int:
        """Accumulate the clock (gain 3) waveforms of a list or :class:`~claudland.wfcomp.WaveformBatch`."""
        from .wfcomp import WaveformBatch
        b = WaveformBatch.from_list(waveforms)
        b = b.select(b.gain == 3)
        if len(b) == 0:
            return 0
        per = self.period_samples(b.samples.astype(np.float64), self.min_period, self.max_period)
        ok = np.isfinite(per) & (b.cable < self._sum.shape[0])
        np.add.at(self._sum, (b.cable[ok], b.atwd[ok]), per[ok])
        np.add.at(self._n, (b.cable[ok], b.atwd[ok]), 1)
        return int(ok.sum())

    @property
    def n_measured(self) -> int:
        """Number of (cable, ATWD) channels with a measured clock period."""
        return int((self._n > 0).sum())

    def period(self) -> np.ndarray:
        """Mean clock period in samples per (cable, ATWD); NaN where not measured."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self._n > 0, self._sum / np.maximum(self._n, 1), np.nan)

    def bin_ns(self, default: float = DEFAULT_BIN_NS) -> np.ndarray:
        """ATWD sampling period in ns per (cable, ATWD), *default* where not measured."""
        p = self.period()
        out = np.full(p.shape, default)
        ok = np.isfinite(p) & (p > 0)
        out[ok] = CLOCK_PERIOD_NS / p[ok]
        return out

    def apply(self, calib: TQCalibration) -> None:
        """Store the measured bin widths in *calib*."""
        calib.bin_ns[:] = self.bin_ns()
        calib.meta["clock_channels"] = self.n_measured
