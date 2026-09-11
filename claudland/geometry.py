"""PMT geometry and cable-number conventions.

Cable numbers (``AKatDefs.hh``)::

    0    .. 1324  inner detector 17-inch PMTs  (1325)
    1325 .. 1878  inner detector 20-inch PMTs  (554)
    1879 .. 2119  outer detector 20-inch PMTs
    2120 .. 2125  5-inch chimney PMTs

Positions come from ``pmt_xyz.dat`` (Kat constants: cable, x, y, z in **cm**,
z pointing up along the chimney axis; identical to ``Kat/src/pmt_xyz.cc``).
The inner PMTs are listed on the 850 cm sphere to which they are bolted; the
photocathode / first dynode where the photo-electron is produced sits ~20 cm
further in, which is why Kat's ``KPmtTable`` rescales the positions by 830/850.
:meth:`PMTTable.load` exposes this as ``radius_scale`` (applied to the inner
tubes only).  The balloon holding the liquid scintillator has radius 650 cm.

All inner tubes are Hamamatsu 20-inch (R3600-type) envelopes.  The "17-inch"
tubes (cables 0-1324) are the fast-timing version whose photocathode is
masked at the edges to the area of a 17-inch tube; the "20-inch" tubes
(cables 1325-1878) use the full photocathode.  Hence the same position table
and the two photocathode areas below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Optional

import numpy as np

__all__ = ["PMTTable", "N_ID17", "N_ID20", "N_ID", "N_CABLES", "BALLOON_RADIUS_CM",
           "PMT_RADIUS_CM", "CABLE_ID17", "CABLE_ID20", "CABLE_OD", "CABLE_5INCH"]

N_ID17 = 1325
N_ID20 = 554
N_ID = N_ID17 + N_ID20          # 1879
N_CABLES = 2126                 # rows in pmt_xyz.dat
BALLOON_RADIUS_CM = 650.0
PMT_RADIUS_CM = 850.0
PMT_RADIUS_LIGHTSHIELD_CM = 850.0

CABLE_ID17 = range(0, N_ID17)
CABLE_ID20 = range(N_ID17, N_ID)
CABLE_OD = range(N_ID, 2120)
CABLE_5INCH = range(2120, 2126)

# Photocathode areas (cm^2) used for the relative solid angle of 17" vs 20" tubes:
# effective radii 21.83 cm (masked) and 23.0 cm (full) -- AKatDefs
# kIDPMT17/20PhotoCathodeCylindricalRadius.
AREA_17_CM2 = np.pi * 21.83 ** 2
AREA_20_CM2 = np.pi * 23.0 ** 2



@dataclass
class PMTTable:
    """PMT positions and types, indexed by cable number."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    radius_scale: float = 1.0

    @classmethod
    def load(cls, path: Optional[str] = None, radius_scale: float = 1.0) -> "PMTTable":
        """Load the table; ``radius_scale`` (e.g. 830/850) moves the inner tubes radially
        from the mounting sphere to the effective photocathode position."""
        if path is None:
            from . import config
            path = str(config.get().require("pmt_table"))
        arr = np.loadtxt(path)
        cable = arr[:, 0].astype(int)
        n = int(cable.max()) + 1
        x = np.full(n, np.nan); y = np.full(n, np.nan); z = np.full(n, np.nan)
        x[cable] = arr[:, 1]; y[cable] = arr[:, 2]; z[cable] = arr[:, 3]
        if radius_scale != 1.0:
            inner = cable < N_ID
            for a in (x, y, z):
                a[cable[inner]] *= radius_scale
        return cls(x, y, z, radius_scale)

    def __len__(self) -> int:
        return len(self.x)

    @property
    def xyz(self) -> np.ndarray:
        """``(n, 3)`` array of positions in cm."""
        return np.stack([self.x, self.y, self.z], axis=1)

    @property
    def r(self) -> np.ndarray:
        """Radius of every tube (cm)."""
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @property
    def cable(self) -> np.ndarray:
        """Cable numbers (row index of the table)."""
        return np.arange(len(self.x))

    # -- type masks -----------------------------------------------------------
    @property
    def is_id17(self) -> np.ndarray:
        """Mask of the 17-inch inner-detector tubes (cables 0-1324)."""
        c = self.cable
        return c < N_ID17

    @property
    def is_id20(self) -> np.ndarray:
        """Mask of the 20-inch inner-detector tubes (cables 1325-1878)."""
        c = self.cable
        return (c >= N_ID17) & (c < N_ID)

    @property
    def is_id(self) -> np.ndarray:
        """Mask of all inner-detector tubes (cables 0-1878)."""
        return self.cable < N_ID

    @property
    def is_od(self) -> np.ndarray:
        """Mask of the outer-detector tubes (cables 1879-2119)."""
        c = self.cable
        return (c >= N_ID) & (c < 2120)

    @property
    def area(self) -> np.ndarray:
        """Photocathode area (cm^2) per cable (17" or 20"; OD treated as 20")."""
        return np.where(self.is_id17, AREA_17_CM2, AREA_20_CM2)

    @property
    def normal(self) -> np.ndarray:
        """Unit vector from PMT toward the detector centre (PMT axis)."""
        p = self.xyz
        d = np.linalg.norm(p, axis=1)
        d[d == 0] = 1.0
        return -p / d[:, None]

    @staticmethod
    def tube_type(cable) -> np.ndarray:
        """0 = 17", 1 = 20", 2 = OD, 3 = 5"."""
        c = np.asarray(cable)
        return np.select([c < N_ID17, c < N_ID, c < 2120], [0, 1, 2], default=3)
