"""Visible-energy <-> real-energy conversion of the KamLAND analysis (``ParticleEnergy`` tables).

The scintillator light is not proportional to the deposited energy: ionisation
quenching (Birks) removes light at low energy and Čerenkov emission adds some
at high energy (Detwiler §4.4.4).  The collaboration tabulated the ratio
``E_vis / E_real`` for gammas, electrons and positrons with a Monte Carlo fitted
to the calibration sources (``KVFParticleEnergy`` of the vector-file code, tables
``$KAMLAND_CONST_DIR/vf/ParticleEnergy/{Gamma,Electron,Positron}.table``).  Each
table line is ``E_real  ratio_best  ratio_err1 ... ratio_err6`` (six systematic
variants); the positron table is for the *total* positron energy including the
two annihilation gammas, so its 1.022 MeV row is the ⁶⁸Ge source.  Interpolation
is the natural cubic spline of the original code (``KVFSpline``).

The tables are private to the collaboration and are read from the directory named
by ``particle_energy`` in ``claudland.toml`` (default ``private/ParticleEnergy``).

Example::

    pe = ParticleEnergy.load()
    pe.gamma_visible(1.3325)          # 1.254 MeV
    pe.source_visible_energy("source-60Co")   # 2.343 MeV (1.173 + 1.333 MeV gammas)
    pe.source_visible_energy("source-68Ge")   # 0.846 MeV (two 0.511 MeV gammas)
    pe.visible_to_gamma(2.343)        # 2.506 MeV
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["ParticleEnergy", "NaturalSpline", "SOURCE_GAMMA_LINES_MEV", "source_gamma_lines"]

#: gamma lines deposited by the calibration sources (MeV); a positron source counts its two annihilation gammas
SOURCE_GAMMA_LINES_MEV: Dict[str, List[float]] = {
    "60Co": [1.17324, 1.3325],
    "68Ge": [0.511, 0.511],
    "65Zn": [1.11552],
    "203Hg": [0.27919],
}


def source_gamma_lines(run_type: str) -> Optional[List[float]]:
    """Gamma lines of the source named in a run type such as ``'source-68Ge'`` (None if unknown)."""
    for iso, lines in SOURCE_GAMMA_LINES_MEV.items():
        if iso.lower() in (run_type or "").lower():
            return list(lines)
    return None


class NaturalSpline:
    """Natural cubic spline through (x, y) (second derivative zero at both ends), the
    ``KVFSpline`` of the KamLAND vector-file code.  Extrapolates with the end cubics."""

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        order = np.argsort(x)
        self.x = x[order]; self.y = y[order]
        n = len(x)
        y2 = np.zeros(n); u = np.zeros(n)
        for i in range(1, n - 1):
            sig = (self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            u[i] = ((self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
                    - (self.y[i] - self.y[i - 1]) / (self.x[i] - self.x[i - 1]))
            u[i] = (6.0 * u[i] / (self.x[i + 1] - self.x[i - 1]) - sig * u[i - 1]) / p
        for k in range(n - 2, -1, -1):
            y2[k] = y2[k] * y2[k + 1] + u[k]
        self.y2 = y2

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        k = np.clip(np.searchsorted(self.x, t, side="right") - 1, 0, len(self.x) - 2)
        h = self.x[k + 1] - self.x[k]
        a = (self.x[k + 1] - t) / h
        b = (t - self.x[k]) / h
        return (a * self.y[k] + b * self.y[k + 1]
                + ((a ** 3 - a) * self.y2[k] + (b ** 3 - b) * self.y2[k + 1]) * h * h / 6.0)


@dataclass
class _Table:
    e_real: np.ndarray
    ratio: np.ndarray          #: E_vis / E_real, best value
    variants: np.ndarray       #: (n, 6) systematic variants of the ratio

    @property
    def e_vis(self) -> np.ndarray:
        return self.e_real * self.ratio

    def forward(self, variant: int = 0) -> NaturalSpline:
        r = self.ratio if variant == 0 else self.variants[:, variant - 1]
        return NaturalSpline(self.e_real, r)

    def inverse(self, variant: int = 0) -> NaturalSpline:
        r = self.ratio if variant == 0 else self.variants[:, variant - 1]
        return NaturalSpline(self.e_real * r, 1.0 / r)


class ParticleEnergy:
    """E_vis <-> E_real for gammas, electrons and positrons from the KamLAND tables.

    ``variant`` 0 is the best fit; 1..6 select the systematic variants of the
    tables (their spread is the energy-scale uncertainty of the analysis)."""

    FILES = {"gamma": "Gamma.table", "electron": "Electron.table", "positron": "Positron.table"}

    def __init__(self, tables: Dict[str, _Table], source: Optional[str] = None):
        self.tables = tables
        self.source = source
        self._fwd: Dict[tuple, NaturalSpline] = {}
        self._inv: Dict[tuple, NaturalSpline] = {}

    @classmethod
    def load(cls, directory: Optional[str] = None) -> "ParticleEnergy":
        """Read the tables from *directory* or from ``particle_energy`` of ``claudland.toml``."""
        if directory is None:
            from . import config
            directory = str(config.get().require("particle_energy"))
        tables = {}
        for key, name in cls.FILES.items():
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"particle-energy table {path} not found")
            arr = np.loadtxt(path)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(f"unexpected format of {path}")
            var = arr[:, 2:8] if arr.shape[1] >= 8 else np.repeat(arr[:, 1:2], 6, axis=1)
            tables[key] = _Table(arr[:, 0], arr[:, 1], var)
        return cls(tables, directory)

    @classmethod
    def available(cls) -> bool:
        """True if the tables exist at the configured location."""
        from . import config
        d = config.get().particle_energy
        return all(os.path.exists(os.path.join(d, n)) for n in cls.FILES.values())

    # -- generic --------------------------------------------------------------------
    def visible(self, particle: str, e_real, variant: int = 0):
        """E_vis (MeV) of a *particle* ('gamma', 'electron', 'positron') of real energy *e_real*."""
        key = (particle, variant)
        if key not in self._fwd:
            self._fwd[key] = self.tables[particle].forward(variant)
        e = np.asarray(e_real, dtype=float)
        return e * self._fwd[key](e)

    def real(self, particle: str, e_vis, variant: int = 0):
        """Real energy (MeV) of a *particle* that produced the visible energy *e_vis*."""
        key = (particle, variant)
        if key not in self._inv:
            self._inv[key] = self.tables[particle].inverse(variant)
        e = np.asarray(e_vis, dtype=float)
        return e * self._inv[key](e)

    # -- convenience ------------------------------------------------------------------
    def gamma_visible(self, e, variant: int = 0):
        return self.visible("gamma", e, variant)

    def electron_visible(self, e, variant: int = 0):
        return self.visible("electron", e, variant)

    def positron_visible(self, e, variant: int = 0):
        """E_vis of a positron of *total* energy *e* (kinetic + 1.022 MeV annihilation)."""
        return self.visible("positron", e, variant)

    def visible_to_gamma(self, e_vis, variant: int = 0):
        return self.real("gamma", e_vis, variant)

    def visible_to_electron(self, e_vis, variant: int = 0):
        return self.real("electron", e_vis, variant)

    def visible_to_positron(self, e_vis, variant: int = 0):
        return self.real("positron", e_vis, variant)

    def source_visible_energy(self, run_type: str, variant: int = 0) -> Optional[float]:
        """Visible energy of a calibration source (sum over its gamma lines), or None if unknown."""
        lines = source_gamma_lines(run_type)
        if lines is None:
            return None
        return float(np.sum(self.gamma_visible(np.asarray(lines), variant)))
