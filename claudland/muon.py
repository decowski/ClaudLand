"""Muon track reconstruction for muons crossing the inner detector.

Model
-----
A muon enters the PMT sphere (radius ``R_PMT`` = 850 cm) at ``e`` at time
``t0``, travels with speed *c* along the unit vector ``u`` and leaves at
``x = e + L u``.  Light is emitted all along the track (Cherenkov light in
the buffer oil, scintillation + Cherenkov light in the scintillator) and
travels with the effective speed *c/n*.  The *earliest* light that can reach
PMT *i* comes from the point of the track that sees the PMT under the
Cherenkov angle ``cos θc = 1/n``; with ``z`` the along-track and ``ρ`` the
perpendicular coordinate of the PMT in the track frame::

    t_i = t0 + (z_i + ρ_i tan θc) / c           (emission point inside the track)
    t_i = t0 + n |P_i − e| / c                   (emission point before the entrance)
    t_i = t0 + L / c + n |P_i − x| / c           (emission point beyond the exit)

This is the "cone fit" of ``Kat/src/KatMuonFitter.cc`` (Tajima, Mitsui
2002–2003) written as an explicit 5-parameter least-squares problem: two
points on the PMT sphere (2 × 2 tangent-plane coordinates) and ``t0``,
solved by damped Gauss-Newton with a shrinking time window.  The
effective index *n* is either fixed, the impact-parameter-dependent Kat
table (1.65 for tracks through the centre → 1.4 for buffer-oil tracks,
which absorbs the scintillation rise time), or fitted per event as a sixth
parameter.

Hit preparation
---------------
Through-going muons saturate the high-gain (and often the medium-gain)
waveforms of every tube.  :func:`muon_hits` therefore takes, per cable, the
*time* from the earliest pulse and the *charge* from the highest-gain
waveform that is not saturated (low gain as last resort).

Outer-detector hits do not constrain the entry point (the OD light is
reflected and diffuse, arriving ~60 ns late with a 70 ns spread), but their
*ordering* in time follows the muon: on well-measured LS tracks the OD hit
time grows with the along-track coordinate of the tube with slope ≈ 1/c and
the entrance-side tubes fire first in 97 % of the tracks.  The fitter
therefore scores each candidate with an OD timing term
``mean(min(((t_OD − t0 − z/c − offset)/σ_OD)², cap))`` (offset profiled), which
fixes the orientation of short buffer-oil clippers.

Chimney muons: the six 5-inch tubes at the top of the chimney (cables
2120-2125) see the scintillator in the chimney.  Following
``KatMuonFitter::checkChimney`` a muon with ``Q_5inch >= 100 p.e.`` is flagged
as a chimney muon; it gets an extra seed with the entrance at the charge
centroid of the ID hits above z = 800 cm and a score penalty when the fitted
track passes more than ``chimney_radius`` from the top of the PMT sphere.
The closest approach of every track to the chimney point is stored.

Track lengths follow Kat: LS length = chord through r = 650 cm, buffer-oil
length = chord through r = 830 cm minus the LS length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import PMTTable, N_ID, N_ID17, BALLOON_RADIUS_CM

__all__ = ["C_CM_NS", "R_PMT", "R_BO", "R_LS", "DQDX_LS", "DQDX_BO", "CHIMNEY_POINT", "MUHIT_DTYPE", "MUON_DTYPE", "MuonTrack",
           "MuonTrackFitter", "MuonChargeModel",
           "muon_hits", "is_muon", "n200_od", "muon_row", "first_light_time", "chord_length", "kat_index"]

C_CM_NS = 29.9792458          #: muon speed (cm/ns)
R_PMT = 850.0                 #: entrance/exit sphere (PMT bolting sphere, as in Kat)
R_BO = 830.0                  #: outer boundary of the light-producing buffer oil (photocathode radius)
R_LS = BALLOON_RADIUS_CM      #: balloon radius
R_TANK = 900.0
DQDX_LS = 629.4               #: 17-inch charge per cm of track in the scintillator (Ichimura, Abe, Keefer)
DQDX_BO = 31.45               #: 17-inch Cherenkov charge per cm of track in the buffer oil
CHIMNEY_POINT = np.array([0.0, 0.0, R_PMT])   #: top of the PMT sphere, below the chimney
CHIMNEY_Z = 800.0             #: ID hits above this z define the chimney entrance (Kat)

MUHIT_DTYPE = np.dtype([
    ("cable", np.int16), ("t", np.float32), ("q", np.float32), ("gain", np.int8),
    ("saturated", np.bool_), ("t_lead", np.float32),
])

MUON_DTYPE = np.dtype([
    ("index", np.int32), ("event", np.int32), ("run", np.int32), ("unix_time", np.float64),
    ("timestamp", np.int64), ("trigger", np.int64),
    ("nhit", np.int16), ("nhit_od", np.int16), ("n200_od", np.int16), ("nsat", np.int16),
    ("q17", np.float32), ("q20", np.float32), ("q_od", np.float32), ("q_5inch", np.float32),
    ("ok", np.bool_), ("converged", np.bool_), ("n_iter", np.int16), ("n_used", np.int16), ("frac_used", np.float32),
    ("sigma_t", np.float32), ("chi2", np.float32),
    ("ex", np.float32), ("ey", np.float32), ("ez", np.float32),       # entrance on r = 850
    ("xx", np.float32), ("xy", np.float32), ("xz", np.float32),       # exit on r = 850
    ("ux", np.float32), ("uy", np.float32), ("uz", np.float32),       # direction (entrance -> exit)
    ("t0", np.float32), ("cos_zenith", np.float32), ("azimuth", np.float32),
    ("impact", np.float32), ("l_ls", np.float32), ("l_bo", np.float32), ("n_eff", np.float32),
    ("seed", np.int8), ("score", np.float32), ("q_expected", np.float32), ("delta_q", np.float32),
    ("pattern_score", np.float32), ("charge_scale", np.float32), ("od_score", np.float32), ("n_od_used", np.int16),
    ("chimney", np.bool_), ("chimney_dist", np.float32),
])


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def unit(v: np.ndarray) -> np.ndarray:
    """Unit vector along *v*."""
    return v / np.linalg.norm(v)


def tangent_basis(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to *p* (and to each other)."""
    n = unit(p)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(a - np.dot(a, n) * n)
    e2 = np.cross(n, e1)
    return e1, e2


def impact_parameter(e: np.ndarray, u: np.ndarray) -> float:
    """Distance of the line ``e + s u`` from the detector centre."""
    return float(np.linalg.norm(e - np.dot(e, u) * u))


def chord_length(b: float, R: float) -> float:
    """Length of the chord of a sphere of radius *R* at impact parameter *b*."""
    return 2.0 * np.sqrt(R * R - b * b) if b < R else 0.0


def sphere_crossings(e: np.ndarray, u: np.ndarray, R: float) -> Optional[Tuple[float, float]]:
    """Path parameters ``s`` where ``e + s u`` crosses the sphere of radius *R* (``s1 <= s2``)."""
    bb = np.dot(e, u)
    cc = np.dot(e, e) - R * R
    dd = bb * bb - cc
    if dd <= 0:
        return None
    sq = np.sqrt(dd)
    return -bb - sq, -bb + sq


def kat_index(impact: float) -> float:
    """Effective refractive index vs impact parameter (``KatMuonFitter::makeTrackFromIDt``)."""
    if impact < 400.0:
        return 1.65
    if impact < 650.0:
        return 1.65 + (1.5 - 1.65) * (impact - 400.0) / 250.0
    if impact < 690.0:
        return 1.5 + (1.45 - 1.5) * (impact - 650.0) / 40.0
    if impact < 730.0:
        return 1.45 + (1.4 - 1.45) * (impact - 690.0) / 40.0
    return 1.4


def fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    """*n* approximately uniform points on a sphere ``[n, 3]``."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return radius * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def first_light_time(P: np.ndarray, e: np.ndarray, u: np.ndarray, L: float, t0: float, n: float,
                     c: float = C_CM_NS) -> np.ndarray:
    """Earliest possible light arrival time at the points *P* ``[m, 3]`` (see module docstring)."""
    d = P - e
    z = d @ u
    rho2 = np.einsum("ij,ij->i", d, d) - z * z
    rho = np.sqrt(np.clip(rho2, 0.0, None))
    tan = np.sqrt(n * n - 1.0)
    zem = z - rho / tan
    t_cone = t0 + (z + rho * tan) / c
    t_ent = t0 + n * np.sqrt(rho2 + z * z) / c
    dx = P - (e + L * u)
    t_exit = t0 + L / c + n * np.sqrt(np.einsum("ij,ij->i", dx, dx)) / c
    return np.where(zem < 0.0, t_ent, np.where(zem > L, t_exit, t_cone))


# ---------------------------------------------------------------------------
# per-tube hit preparation
# ---------------------------------------------------------------------------
def muon_hits(hits: np.ndarray, time_key: str = "t") -> np.ndarray:
    """Collapse a HIT_DTYPE array (all gains, ID + OD) to one entry per cable.

    time  : ``time_key`` of the earliest found high-gain pulse (or the earliest
            pulse of any gain if the high gain is missing);
    charge: from the highest-gain waveform of that cable that is not saturated,
            else the lowest-gain one available.
    """
    ok = np.isfinite(hits["t"]) & (hits["q"] > 0)
    h = hits[ok]
    if len(h) == 0:
        return np.zeros(0, dtype=MUHIT_DTYPE)
    # earliest waveform per (cable, gain)
    order = np.lexsort((h["t"], h["gain"], h["cable"]))
    h = h[order]
    key = h["cable"].astype(np.int64) * 4 + h["gain"]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    h = h[first]
    cables, start = np.unique(h["cable"], return_index=True)
    out = np.zeros(len(cables), dtype=MUHIT_DTYPE)
    out["cable"] = cables
    end = np.append(start[1:], len(h))
    for k, (a, b) in enumerate(zip(start, end)):
        grp = h[a:b]                       # sorted by gain (0, 1, 2)
        # time from the highest gain present (earliest pulse)
        out["t"][k] = grp[time_key][0]
        out["t_lead"][k] = grp["t_lead"][0]
        uns = np.flatnonzero(~grp["saturated"])
        j = uns[0] if len(uns) else len(grp) - 1
        out["q"][k] = grp["q"][j]
        out["gain"][k] = grp["gain"][j]
        out["saturated"][k] = grp["saturated"][j]
    return out


def n200_od(t_od: np.ndarray, window: float = 200.0) -> int:
    """Maximum number of OD hits inside a sliding window of *window* ns (``N200_OD`` of the theses)."""
    t = np.sort(np.asarray(t_od, dtype=np.float64))
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return 0
    j = np.searchsorted(t, t + window, side="right")
    return int((j - np.arange(len(t))).max())


def is_muon(q17: float, n_od: int, q17_min: float = 500.0, q17_alone: float = 10000.0, n_od_min: int = 5) -> bool:
    """The standard KamLAND muon selection: ``(N200_OD >= 5 and Q17 >= 500 p.e.) or Q17 >= 10000 p.e.``

    *n_od* should be the 200 ns coincidence count :func:`n200_od`; the total OD
    multiplicity lets calibration-source events with accidental OD hits through."""
    return (n_od >= n_od_min and q17 >= q17_min) or q17 >= q17_alone


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class MuonTrack:
    """A fitted muon track: entrance and exit on the PMT sphere, timing, quality figures and derived geometry."""
    entrance: np.ndarray            #: on r = R_PMT (cm)
    exit: np.ndarray                #: on r = R_PMT (cm)
    t0: float                       #: time at the entrance (ns, trigger frame)
    n_eff: float
    converged: bool
    n_iter: int
    n_used: int
    n_hits: int
    sigma_t: float                  #: rms of the residuals of the hits inside the final window (ns)
    chi2: float
    seed: int = 0
    score: float = np.nan           #: used-hit fraction minus charge penalties (higher is better)
    q17: float = np.nan
    q_expected: float = np.nan      #: minimum-ionising 17-inch charge for this geometry
    pattern_score: float = np.nan   #: mean capped squared log-charge residual of the per-tube pattern
    charge_scale: float = np.nan    #: observed / minimum-ionising charge from the pattern fit
    od_score: float = np.nan        #: OD timing term (capped mean squared residual in units of σ_OD²)
    n_od_used: int = 0
    chimney: bool = False           #: Q_5inch above threshold (muon through the chimney)
    residuals: Optional[np.ndarray] = None   #: per input hit (ns)
    used: Optional[np.ndarray] = None

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from entrance to exit."""
        return unit(self.exit - self.entrance)

    @property
    def length(self) -> float:
        """Distance between entrance and exit (cm)."""
        return float(np.linalg.norm(self.exit - self.entrance))

    @property
    def impact(self) -> float:
        """Impact parameter: distance of the track line from the detector centre (cm)."""
        return impact_parameter(self.entrance, self.direction)

    @property
    def l_ls(self) -> float:
        """Track length in the scintillator (chord through r = 650 cm)."""
        return chord_length(self.impact, R_LS)

    @property
    def l_bo(self) -> float:
        """Track length in the buffer oil (chord through r = 830 cm minus the LS length)."""
        return chord_length(self.impact, R_BO) - self.l_ls

    @property
    def cos_zenith(self) -> float:
        """cos of the zenith angle of the muon direction (1 = vertically downward)."""
        return float(-self.direction[2])

    @property
    def azimuth(self) -> float:
        """Azimuth of the direction (degrees, from +x towards +y)."""
        u = self.direction
        return float(np.degrees(np.arctan2(u[1], u[0])))

    @property
    def frac_used(self) -> float:
        """Fraction of the fitted tubes inside the final time window."""
        return self.n_used / self.n_hits if self.n_hits else 0.0

    @property
    def chimney_dist(self) -> float:
        """Closest approach of the track line to the top of the PMT sphere (below the chimney)."""
        return float(self.distance(CHIMNEY_POINT)[0])

    def distance(self, p: np.ndarray) -> np.ndarray:
        """Perpendicular distance of point(s) *p* from the track line."""
        p = np.atleast_2d(p)
        d = p - self.entrance
        z = d @ self.direction
        return np.sqrt(np.clip(np.einsum("ij,ij->i", d, d) - z * z, 0, None))

    def summary(self) -> str:
        """One-line summary for printing."""
        e, x = self.entrance, self.exit
        return (f"entrance ({e[0]:.0f},{e[1]:.0f},{e[2]:.0f}) exit ({x[0]:.0f},{x[1]:.0f},{x[2]:.0f}) "
                f"cosZ={self.cos_zenith:.2f} b={self.impact:.0f} L_LS={self.l_ls:.0f} L_BO={self.l_bo:.0f} cm "
                f"n={self.n_eff:.3f} sigma_t={self.sigma_t:.2f} ns used {self.n_used}/{self.n_hits} "
                f"Q17/Qexp={self.q17 / self.q_expected if self.q_expected else np.nan:.2f} pat={self.pattern_score:.2f} od={self.od_score:.2f}{' CHIMNEY d=%.0f' % self.chimney_dist if self.chimney else ''} score={self.score:.2f} "
                f"{'conv' if self.converged else 'NOT conv'} ({self.n_iter} it)")


# ---------------------------------------------------------------------------
# charge model
# ---------------------------------------------------------------------------
class MuonChargeModel:
    """Expected charge per 17-inch tube for a minimum-ionising muon track.

    Two components, both normalised to the standard KamLAND yields:

    * scintillation: ``dqdx_ls`` p.e. per cm of track in the scintillator,
      emitted isotropically and collected with the point-source acceptance
      ``A_i (0.1 + 0.9 cos θ_i) e^{-d/Λ} / d²`` (Detwiler Eq. 4.8), integrated
      along the LS chord in steps of ``ds``;
    * Cherenkov: ``dqdx_bo`` p.e. per cm along the whole track inside
      r < 830 cm.  A fraction ``1 - f_iso`` is emitted on the cone
      ``cos θc = 1/n`` (tube *i* receives it from the emission point that sees it
      under the Cherenkov angle, spread over the ring circumference, ∝ 1/d);
      the fraction ``f_iso`` (default 0.5) is treated as isotropic to mimic
      scattered and reflected light, which dominates the diffuse "blob" seen
      from buffer-oil clippers.

    :meth:`residual` compares measured and expected charges in log space after
    profiling a global scale factor (showers, δ-rays and gain drifts change the
    amount of light, not its pattern), with a Huber-type cap on the residuals.
    """

    def __init__(self, pmts: PMTTable, dqdx_ls: float = DQDX_LS, dqdx_bo: float = DQDX_BO, index: float = 1.5,
                 attenuation_cm: float = 2500.0, ds: float = 25.0, q0: float = 3.0, cap: float = 1.5,
                 live: Optional[np.ndarray] = None, f_iso: float = 0.5):
        self.P = pmts.xyz[:N_ID17].astype(np.float64)
        self.area = pmts.area[:N_ID17].astype(np.float64)
        self.normal = pmts.normal[:N_ID17].astype(np.float64)
        self.dqdx_ls = dqdx_ls
        self.dqdx_bo = dqdx_bo
        self.n = index
        self.lam = attenuation_cm
        self.ds = ds
        self.q0 = q0
        self.cap = cap
        self.live = np.ones(N_ID17, dtype=bool) if live is None else np.asarray(live[:N_ID17], dtype=bool)
        self.f_iso = f_iso
        self._acc0 = self._acceptance(np.zeros((1, 3)))[0].sum()

    def _acceptance(self, R: np.ndarray) -> np.ndarray:
        """Relative light collection of every tube for point sources at ``R [k, 3]`` -> ``[k, n17]``."""
        dvec = self.P[None, :, :] - R[:, None, :]
        d = np.linalg.norm(dvec, axis=2)
        cos = np.clip(-np.einsum("kij,ij->ki", dvec, self.normal) / d, 0.0, 1.0)
        return self.area[None, :] * (0.1 + 0.9 * cos) * np.exp(-d / self.lam) / (d * d)

    def expected(self, e: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Expected charge (p.e.) of every 17-inch tube for the track e -> x."""
        u = unit(x - e)
        L = float(np.linalg.norm(x - e))
        mu = np.zeros(N_ID17)
        cr = sphere_crossings(e, u, R_LS)
        if cr is not None:
            s0, s1 = max(cr[0], 0.0), min(cr[1], L)
            if s1 > s0:
                ss = np.arange(s0 + self.ds / 2, s1, self.ds)
                if len(ss):
                    acc = self._acceptance(e[None, :] + ss[:, None] * u[None, :])
                    mu += self.dqdx_ls / self._acc0 * acc.sum(0) * (s1 - s0) / len(ss)
        cr2 = sphere_crossings(e, u, R_BO)
        if cr2 is not None:
            a0, a1 = max(cr2[0], 0.0), min(cr2[1], L)
            if a1 > a0:
                if self.f_iso > 0:      # scattered / reflected Cherenkov light: isotropic along the track
                    ss = np.arange(a0 + self.ds / 2, a1, self.ds)
                    if len(ss):
                        acc = self._acceptance(e[None, :] + ss[:, None] * u[None, :])
                        mu += self.f_iso * self.dqdx_bo / self._acc0 * acc.sum(0) * (a1 - a0) / len(ss)
                d = self.P - e
                z = d @ u
                rho = np.sqrt(np.clip(np.einsum("ij,ij->i", d, d) - z * z, 0, None))
                tan = np.sqrt(self.n * self.n - 1.0)
                sin = tan / self.n
                zem = z - rho / tan
                inside = (zem > a0) & (zem < a1)
                if inside.any():
                    em = e[None, :] + zem[:, None] * u[None, :]
                    dv = self.P - em
                    dist = np.maximum(np.linalg.norm(dv, axis=1), 1.0)
                    cosinc = np.clip(-np.einsum("ij,ij->i", dv, self.normal) / dist, 0.0, 1.0)
                    g = self.area * (0.1 + 0.9 * cosinc) * np.exp(-dist / self.lam) / (2 * np.pi * dist * sin)
                    g = np.where(inside, g, 0.0)
                    mu += (1.0 - self.f_iso) * self.dqdx_bo * (a1 - a0) * g / g.sum()
        return mu

    def mask(self, q: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Tubes that enter the charge comparison: live and either hit or expected to be hit."""
        return self.live & ((q > 0.5) | (mu > 0.5))

    def residual(self, q: np.ndarray, mu: np.ndarray, ok: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Log-charge residuals ``ln(q+q0) - ln(mu+q0) - s`` for the tubes in *ok* (default
        :meth:`mask`), the IRLS weights (Huber, cap ``cap``) and the profiled scale ``exp(s)``."""
        if ok is None:
            ok = self.mask(q, mu)
        lq = np.log(q[ok] + self.q0)
        lm = np.log(mu[ok] + self.q0)
        s = float(np.median(lq - lm))
        r = lq - lm - s
        w = np.minimum(1.0, self.cap / np.maximum(np.abs(r), 1e-9))
        return r, w, float(np.exp(s))

    def score(self, q: np.ndarray, e: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
        """(mean capped squared log residual, scale) for the track e -> x; lower is better."""
        r, w, scale = self.residual(q, self.expected(e, x))
        if len(r) == 0:
            return np.nan, np.nan
        return float(np.mean(np.minimum(r * r, self.cap ** 2))), scale


# ---------------------------------------------------------------------------
# the fitter
# ---------------------------------------------------------------------------
class MuonTrackFitter:
    """Fit a straight track through the PMT sphere to per-tube first-hit times (and charges).

    Parameters
    ----------
    pmts : PMTTable
    index : effective refractive index for the first-light model: a float (default 1.5),
        ``"kat"`` for the impact-parameter table of ``KatMuonFitter``, or ``"fit"``
        (released as a sixth parameter at the end; ill-determined for central tracks)
    charge_model : :class:`MuonChargeModel` or ``None`` for a time-only fit
    charge_weight : weight λ of the charge term ``λ Σ w_q r_q²`` relative to ``Σ w_t r_t²`` (ns²)
    sigmas : successive Welsch kernel widths (ns) of the robust time weights ``exp(-r²/2σ²)``
    weight : extra per-hit time weights: ``"uniform"``, ``"charge"`` (∝ sqrt(min(q, q_cap))) or ``"kat"`` (1/max(ρ,150))
    seed : ``"scan"`` (global grid search, default), ``"cluster"`` (Kat heuristics) or ``"both"``

    Candidate tracks are ranked by ``frac_used − charge_term(Q17) − pattern_weight × pattern_score``.
    """

    def __init__(self, pmts: PMTTable, index: Union[str, float] = 1.5, charge_model: Optional[MuonChargeModel] = None,
                 charge_weight: float = 10.0, pattern_weight: float = 0.3,
                 sigmas: Sequence[float] = (30.0, 15.0, 8.0, 5.0, 4.0), iters_per_stage: int = 6,
                 weight: str = "uniform", q_min: float = 0.3, q_cap: float = 200.0, used_window: float = 8.0,
                 min_hits: int = 20, c: float = C_CM_NS, tol_cm: float = 1.0, tol_ns: float = 0.1,
                 seed: str = "scan", n_scan: int = 150, scan_window: float = 12.0, scan_index: float = 1.5,
                 n_refine: int = 3, n_candidates: int = 3, n_pattern: int = 20, try_flip: bool = True,
                 dqdx_ls: float = DQDX_LS, dqdx_bo: float = DQDX_BO, q_floor: float = 3000.0,
                 charge_penalty: float = 0.1, charge_log_sigma: float = 0.3,
                 od_weight: float = 0.1, od_sigma: float = 25.0, od_cap: float = 9.0, od_min_hits: int = 4,
                 chimney_q_min: float = 100.0, chimney_radius: float = 200.0, chimney_weight: float = 0.1):
        self.pmts = pmts
        self.chimney_q_min = chimney_q_min
        self.chimney_radius = chimney_radius
        self.chimney_weight = chimney_weight
        self.od_weight = od_weight
        self.od_sigma = od_sigma
        self.od_cap = od_cap
        self.od_min_hits = od_min_hits
        self.xyz = pmts.xyz.astype(np.float64)
        self.index = index
        self.charge_model = charge_model
        self.charge_weight = charge_weight
        self.pattern_weight = pattern_weight
        self.sigmas = list(sigmas)
        self.iters_per_stage = iters_per_stage
        self.weight = weight
        self.q_min = q_min
        self.q_cap = q_cap
        self.used_window = used_window
        self.min_hits = min_hits
        self.c = c
        self.tol_cm = tol_cm
        self.tol_ns = tol_ns
        self.seed_mode = seed
        self.n_scan = n_scan
        self.scan_window = scan_window
        self.scan_index = scan_index
        self.n_refine = n_refine
        self.n_candidates = n_candidates
        self.n_pattern = n_pattern
        self.try_flip = try_flip
        self._grid = fibonacci_sphere(n_scan, R_PMT)
        # total-charge consistency: the 17-inch charge of a minimum-ionising muon is
        # dqdx_ls * L_LS + dqdx_bo * L_BO; candidates are penalised by
        # charge_penalty * (log10(Q17 / Q_exp) / charge_log_sigma)^2 (in units of the
        # used-hit fraction).  Together with the per-tube pattern score this breaks the
        # near-degeneracy between a long track through the scintillator and a short
        # track hugging the PMT sphere, which the first-light times alone hardly resolve.
        self.dqdx_ls = dqdx_ls
        self.dqdx_bo = dqdx_bo
        self.q_floor = q_floor
        self.charge_penalty = charge_penalty
        self.charge_log_sigma = charge_log_sigma

    # -- model ------------------------------------------------------------------
    def _n(self, e, u, n_free: Optional[float]) -> float:
        if n_free is not None:
            return n_free
        if isinstance(self.index, (int, float)):
            return float(self.index)
        if self.index == "fit":
            return self.scan_index
        return kat_index(impact_parameter(e, u))

    def predict(self, P: np.ndarray, e: np.ndarray, x: np.ndarray, t0: float, n: float) -> np.ndarray:
        """First-light arrival times at the points *P* for the track e -> x."""
        d = x - e
        L = np.linalg.norm(d)
        return first_light_time(P, e, d / L, L, t0, n, self.c)

    def _weights(self, P, e, x, q) -> np.ndarray:
        if self.weight == "charge":
            return np.sqrt(np.clip(q, 0.0, self.q_cap) / self.q_cap)
        if self.weight == "kat":
            u = unit(x - e)
            d = P - e
            z = d @ u
            rho = np.sqrt(np.clip(np.einsum("ij,ij->i", d, d) - z * z, 0, None))
            return 150.0 / np.maximum(rho, 150.0)
        return np.ones(len(P))

    # -- charge consistency -----------------------------------------------------------
    def expected_charge(self, impact) -> np.ndarray:
        """Minimum-ionising 17-inch charge for tracks at the given impact parameter(s)."""
        b = np.asarray(impact, dtype=np.float64)
        l_ls = np.where(b < R_LS, 2.0 * np.sqrt(np.clip(R_LS ** 2 - b * b, 0, None)), 0.0)
        l_bo = np.where(b < R_BO, 2.0 * np.sqrt(np.clip(R_BO ** 2 - b * b, 0, None)), 0.0) - l_ls
        return np.maximum(self.q_floor, self.dqdx_ls * l_ls + self.dqdx_bo * l_bo)

    def charge_term(self, impact, q17: Optional[float]) -> np.ndarray:
        """Penalty (in units of the used-hit fraction) for the observed Q17 given the track geometry."""
        if q17 is None or not np.isfinite(q17) or q17 <= 0 or self.charge_penalty <= 0:
            return np.zeros(np.shape(impact))
        lr = np.log10(max(q17, 1.0) / self.expected_charge(impact)) / self.charge_log_sigma
        return self.charge_penalty * np.minimum(lr * lr, 100.0)

    def od_term(self, e, x, t0, P_od: Optional[np.ndarray], t_od: Optional[np.ndarray]) -> float:
        """OD timing consistency: capped mean squared residual of ``t_OD - t0 - z/c`` about its median.

        ``z`` is the along-track coordinate of the OD tube.  A track fitted with the
        wrong orientation shows a slope of ``2/c`` in these residuals."""
        if P_od is None or t_od is None or len(t_od) < self.od_min_hits or self.od_weight <= 0:
            return 0.0
        u = unit(x - e)
        z = (P_od - e) @ u
        r = t_od - t0 - z / self.c
        r = r - np.median(r)
        return float(np.mean(np.minimum((r / self.od_sigma) ** 2, self.od_cap)))

    def od_terms(self, E, X, T0, P_od, t_od) -> np.ndarray:
        """Vectorised :meth:`od_term` for candidate arrays ``E, X [k,3]`` and ``T0 [k]``."""
        if P_od is None or t_od is None or len(t_od) < self.od_min_hits or self.od_weight <= 0:
            return np.zeros(len(E))
        D = X - E
        U = D / np.linalg.norm(D, axis=1)[:, None]
        z = np.einsum("kmj,kj->km", P_od[None, :, :] - E[:, None, :], U)
        r = t_od[None, :] - T0[:, None] - z / self.c
        r = r - np.median(r, axis=1, keepdims=True)
        return np.mean(np.minimum((r / self.od_sigma) ** 2, self.od_cap), axis=1)

    def chimney_term(self, E, X) -> np.ndarray:
        """Penalty for chimney muons whose track misses the top of the sphere: ``((d - R)/R)²`` beyond ``R``, capped."""
        E = np.atleast_2d(E); X = np.atleast_2d(X)
        D = X - E
        U = D / np.linalg.norm(D, axis=1)[:, None]
        d = CHIMNEY_POINT[None, :] - E
        z = np.einsum("kj,kj->k", d, U)
        dist = np.sqrt(np.clip(np.einsum("kj,kj->k", d, d) - z * z, 0, None))
        x = np.clip((dist - self.chimney_radius) / self.chimney_radius, 0, None)
        return np.minimum(x * x, 4.0)

    def score(self, track: "MuonTrack", q17: Optional[float], qdense: Optional[np.ndarray] = None,
              P_od: Optional[np.ndarray] = None, t_od: Optional[np.ndarray] = None, chimney: bool = False) -> float:
        """Combined figure of merit of a fitted track (higher is better)."""
        if not np.isfinite(track.t0):
            return -np.inf
        sc = track.frac_used - float(self.charge_term(track.impact, q17))
        if chimney and self.chimney_weight > 0:
            sc -= self.chimney_weight * float(self.chimney_term(track.entrance, track.exit)[0])
        if P_od is not None and t_od is not None and len(t_od) >= self.od_min_hits:
            track.od_score = self.od_term(track.entrance, track.exit, track.t0, P_od, t_od)
            track.n_od_used = len(t_od)
            sc -= self.od_weight * track.od_score
        if self.charge_model is not None and qdense is not None:
            ps, scale = self.charge_model.score(qdense, track.entrance, track.exit)
            track.pattern_score = ps
            track.charge_scale = scale
            if np.isfinite(ps):
                sc -= self.pattern_weight * ps
        return sc

    # -- seeds (Kat heuristics) -----------------------------------------------------------
    @staticmethod
    def _cluster_point(P: np.ndarray, t: np.ndarray, q: np.ndarray, order: np.ndarray, n_cand: int,
                       r_max: float = 200.0, dt_max: float = 10.0, n_nb: int = 2, q_min: float = 10.0) -> Optional[np.ndarray]:
        """Kat's cluster search: the first candidate (in *order*) with >= n_nb neighbours
        within r_max cm and dt_max ns among the following four candidates."""
        cand = order[:max(n_cand, 6)]
        for k in range(len(cand) - 1):
            i = cand[k]
            if q[i] <= q_min:
                continue
            nb = cand[k + 1:k + 5]
            close = (np.linalg.norm(P[nb] - P[i], axis=1) < r_max) & (np.abs(t[nb] - t[i]) < dt_max)
            if close.sum() >= n_nb:
                sel = np.append(nb[close], i)
                w = np.sqrt(np.clip(q[sel], 0, None))
                return (P[sel] * w[:, None]).sum(0) / w.sum()
        return None

    def seeds(self, P: np.ndarray, t: np.ndarray, q: np.ndarray) -> list:
        """Candidate (entrance, exit) pairs on the PMT sphere from the Kat heuristics."""
        out = []
        n = len(P)
        by_time = np.argsort(t)
        ent = self._cluster_point(P, t, q, by_time, n // 10)
        if ent is None:
            ent = P[by_time[:max(5, n // 20)]].mean(0)
        t_ent = np.median(t[by_time[:max(3, n // 50)]])
        late = np.clip(t - t_ent, 1.0, None)
        by_ratio = np.argsort(-q * late)
        ex1 = self._cluster_point(P, t, q, by_ratio, n // 5, q_min=min(100.0, np.percentile(q, 50)))
        by_q = np.argsort(-q)
        ex2 = self._cluster_point(P, t, q, by_q, n // 5, dt_max=1e9, q_min=0.0)
        ent_s = unit(ent) * R_PMT
        for ex in (ex1, ex2):
            if ex is None:
                continue
            ex_s = unit(ex) * R_PMT
            if np.linalg.norm(ex_s - ent_s) > 100.0:
                out.append((ent_s, ex_s))
        if not out:
            out.append((ent_s, -ent_s))
        uniq = []
        for e, x in out:
            if not any(np.linalg.norm(x - x2) < 50.0 for _, x2 in uniq):
                uniq.append((e, x))
        return uniq[:self.n_candidates]

    # -- global scan ----------------------------------------------------------------
    def _scan_score(self, P, t, E, X, n, win) -> Tuple[np.ndarray, np.ndarray]:
        """Score every (entrance, exit) pair of the arrays E, X ``[k, 3]``.

        ``t0`` is profiled out per pair (15th percentile of ``t - t_pred``);
        the score is the number of hits within ``±win`` of the prediction."""
        D = X - E
        L = np.linalg.norm(D, axis=1)
        U = D / L[:, None]
        d = P[None, :, :] - E[:, None, :]
        z = np.einsum("kmj,kj->km", d, U)
        dd = np.einsum("kmj,kmj->km", d, d)
        rho = np.sqrt(np.clip(dd - z * z, 0, None))
        tan = np.sqrt(n * n - 1.0)
        zem = z - rho / tan
        t_cone = (z + rho * tan) / self.c
        t_ent = n * np.sqrt(dd) / self.c
        dx = P[None, :, :] - X[:, None, :]
        t_exit = L[:, None] / self.c + n * np.sqrt(np.einsum("kmj,kmj->km", dx, dx)) / self.c
        pred = np.where(zem < 0, t_ent, np.where(zem > L[:, None], t_exit, t_cone))
        r = t[None, :] - pred
        t0 = np.percentile(r, 15, axis=1)
        score = (np.abs(r - t0[:, None]) < win).sum(axis=1)
        return score, t0

    @staticmethod
    def _impacts(E, X):
        D = X - E
        U = D / np.linalg.norm(D, axis=1)[:, None]
        return np.linalg.norm(E - np.einsum("kj,kj->k", E, U)[:, None] * U, axis=1)

    def scan(self, P, t, q, q17: Optional[float] = None, qdense: Optional[np.ndarray] = None,
             P_od: Optional[np.ndarray] = None, t_od: Optional[np.ndarray] = None, chimney: bool = False) -> list:
        """Coarse global search over pairs of grid points on the PMT sphere.

        Pairs are ranked by the time score (hits within ``scan_window``) minus the
        total-charge term; the best ``n_pattern`` distinct pairs are re-ranked with
        the charge-pattern score, and the best ``n_candidates`` are refined on a
        finer local grid (time + total charge).  Returns ``(entrance, exit, t0)`` triples."""
        G = self._grid
        k = len(G)
        ii, jj = np.triu_indices(k, 1)
        I = np.concatenate([ii, jj]); J = np.concatenate([jj, ii])
        n = self.scan_index
        scores = np.empty(len(I)); t0s = np.empty(len(I))
        step = max(1, 20_000_000 // max(len(P), 1))
        for a in range(0, len(I), step):
            sc, t0 = self._scan_score(P, t, G[I[a:a + step]], G[J[a:a + step]], n, self.scan_window)
            scores[a:a + step] = sc; t0s[a:a + step] = t0
        B = self._impacts(G[I], G[J])
        scores = scores / len(P) - self.charge_term(B, q17)
        order = np.argsort(-scores)
        cands = []
        for o in order:
            e, x = G[I[o]], G[J[o]]
            if all(np.linalg.norm(e - e2) > 150 or np.linalg.norm(x - x2) > 150 for e2, x2, _, _ in cands):
                cands.append((e, x, t0s[o], scores[o]))
            if len(cands) >= self.n_pattern:
                break
        if cands and ((self.charge_model is not None and qdense is not None) or P_od is not None or chimney):
            E = np.array([c[0] for c in cands]); X = np.array([c[1] for c in cands]); T0 = np.array([c[2] for c in cands])
            od = self.od_weight * self.od_terms(E, X, T0, P_od, t_od)
            if chimney and self.chimney_weight > 0:
                od = od + self.chimney_weight * self.chimney_term(E, X)
            rescored = []
            for (e, x, t0, sc), o in zip(cands, od):
                ps = self.charge_model.score(qdense, e, x)[0] if (self.charge_model is not None and qdense is not None) else 0.0
                rescored.append((e, x, t0, sc - self.pattern_weight * (ps if np.isfinite(ps) else 0.0) - o))
            cands = sorted(rescored, key=lambda c: -c[3])
        cands = cands[:self.n_candidates]
        # local refinement of each candidate on finer grids around (e, x)
        out = []
        spacing = 2 * R_PMT * np.sqrt(np.pi / k)
        offs = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], dtype=float)
        for e, x, t0, _ in cands:
            for it in range(self.n_refine):
                h = spacing / (2 ** (it + 1))
                e1, e2 = tangent_basis(e); x1, x2 = tangent_basis(x)
                E = np.array([unit(e + a * h * e1 + b * h * e2) * R_PMT for a, b in offs])
                X = np.array([unit(x + a * h * x1 + b * h * x2) * R_PMT for a, b in offs])
                EE = np.repeat(E, len(X), axis=0); XX = np.tile(X, (len(E), 1))
                sc, t0v = self._scan_score(P, t, EE, XX, n, self.scan_window)
                sc = sc / len(P) - self.charge_term(self._impacts(EE, XX), q17)
                b = int(np.nanargmax(sc))
                e, x, t0 = EE[b], XX[b], t0v[b]
            out.append((e, x, float(t0)))
        return out

    # -- joint robust Gauss-Newton --------------------------------------------------------
    def _loss(self, P, t, w_all, qdense, e, x, t0, n, sig, wt_fixed=None, wq_fixed=None, ok_fixed=None):
        """Robust loss and its ingredients at the parameters (e, x, t0, n)."""
        r_t = t - self.predict(P, e, x, t0, n)
        w_t = w_all * np.exp(-0.5 * (r_t / sig) ** 2) if wt_fixed is None else wt_fixed
        loss = float(np.sum(w_t * r_t * r_t))
        r_q = w_q = ok = None
        if self.charge_model is not None and qdense is not None and self.charge_weight > 0:
            mu = self.charge_model.expected(e, x)
            ok = self.charge_model.mask(qdense, mu) if ok_fixed is None else ok_fixed
            r_q, w_q, _ = self.charge_model.residual(qdense, mu, ok)
            if wq_fixed is not None:
                w_q = wq_fixed
            loss += self.charge_weight * float(np.sum(w_q * r_q * r_q))
        return loss, r_t, w_t, r_q, w_q, ok

    def _fit_from(self, P, t, q, e, x, t0=None, qdense=None) -> "MuonTrack":
        """Damped Gauss-Newton with iteratively re-weighted (Welsch) time residuals and,
        if a charge model is set, Huber-weighted log-charge residuals."""
        release_n = self.index == "fit"
        fit_n = False
        n_free = None
        n_par = 5
        use_q = self.charge_model is not None and qdense is not None and self.charge_weight > 0
        n = self._n(e, unit(x - e), n_free)
        if t0 is None:
            pred0 = self.predict(P, e, x, 0.0, n)
            t0 = float(np.percentile(t - pred0, 15))
        w_all = self._weights(P, e, x, q)
        lam = 1e-3
        n_iter = 0
        step_ok = True
        delta = np.zeros(6)
        h = 1.0     # cm, finite-difference step
        for istage, sig in enumerate(self.sigmas):
            if release_n and istage >= len(self.sigmas) - 2 and not fit_n:
                fit_n = True
                n_par = 6
                n_free = self._n(e, unit(x - e), None)
            for _ in range(self.iters_per_stage):
                n_iter += 1
                n = self._n(e, unit(x - e), n_free)
                loss, r_t, w_t, r_q, w_q, ok_q = self._loss(P, t, w_all, qdense, e, x, t0, n, sig)
                if (w_t > 0.1).sum() < self.min_hits:
                    return self._result(P, t, e, x, t0, n, False, n_iter, seed=0)
                e1, e2 = tangent_basis(e)
                x1, x2 = tangent_basis(x)
                params = [lambda d: (unit(e + d * e1) * R_PMT, x, t0, n), lambda d: (unit(e + d * e2) * R_PMT, x, t0, n),
                          lambda d: (e, unit(x + d * x1) * R_PMT, t0, n), lambda d: (e, unit(x + d * x2) * R_PMT, t0, n),
                          lambda d: (e, x, t0 + d, n)]
                steps = [h, h, h, h, 1.0]
                if fit_n:
                    params.append(lambda d: (e, x, t0, n + d)); steps.append(0.01)
                # Jacobians (numerical) of the time and charge residual vectors
                Jt = np.empty((len(P), n_par)); Jq = np.empty((len(r_q), n_par)) if use_q else None
                for j, (fn, st) in enumerate(zip(params, steps)):
                    ee, xx, tt, nn = fn(st)
                    Jt[:, j] = -(self.predict(P, ee, xx, tt, nn) - (t - r_t)) / st
                    if use_q:
                        if j == 4:
                            Jq[:, j] = 0.0
                        else:
                            rq2, _, _ = self.charge_model.residual(qdense, self.charge_model.expected(ee, xx), ok_q)
                            Jq[:, j] = (rq2 - r_q) / st
                A = Jt.T @ (Jt * w_t[:, None])
                g = Jt.T @ (w_t * r_t)
                if use_q:
                    A += self.charge_weight * (Jq.T @ (Jq * w_q[:, None]))
                    g += self.charge_weight * (Jq.T @ (w_q * r_q))
                step_ok = False
                for _try in range(8):
                    try:
                        delta = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue
                    smax = np.max(np.abs(delta[:4]))
                    if smax > 200.0:
                        delta *= 200.0 / smax
                    e_new = unit(e + delta[0] * e1 + delta[1] * e2) * R_PMT
                    x_new = unit(x + delta[2] * x1 + delta[3] * x2) * R_PMT
                    t0_new = t0 + delta[4]
                    n_new = float(np.clip(n_free + delta[5], 1.2, 2.2)) if fit_n else n_free
                    if np.linalg.norm(x_new - e_new) < 50.0:
                        lam *= 10
                        continue
                    nn = self._n(e_new, unit(x_new - e_new), n_new)
                    loss_new = self._loss(P, t, w_all, qdense, e_new, x_new, t0_new, nn, sig, wt_fixed=w_t, wq_fixed=w_q, ok_fixed=ok_q)[0]
                    if loss_new <= loss:
                        e, x, t0, n_free = e_new, x_new, t0_new, n_new
                        lam = max(lam / 3.0, 1e-6)
                        step_ok = True
                        break
                    lam *= 10
                if not step_ok:
                    break
                if np.max(np.abs(delta[:4])) < self.tol_cm and abs(delta[4]) < self.tol_ns:
                    break
        n = self._n(e, unit(x - e), n_free)
        return self._result(P, t, e, x, t0, n, step_ok, n_iter, seed=0)

    def _result(self, P, t, e, x, t0, n, converged, n_iter, seed) -> "MuonTrack":
        r = t - self.predict(P, e, x, t0, n)
        used = np.abs(r) < self.used_window
        ru = r[used]
        sigma = float(np.sqrt(np.mean(ru * ru))) if len(ru) else np.nan
        return MuonTrack(entrance=e, exit=x, t0=float(t0), n_eff=float(n), converged=bool(converged and used.sum() >= self.min_hits),
                         n_iter=n_iter, n_used=int(used.sum()), n_hits=len(P), sigma_t=sigma, chi2=float(np.sum(ru * ru)),
                         seed=seed, residuals=r, used=used)

    # -- public entry point -----------------------------------------------------------
    def fit(self, cable: np.ndarray, t: np.ndarray, q: np.ndarray, q17: Optional[float] = None,
            use_20inch: bool = False, cable_od: Optional[np.ndarray] = None, t_od: Optional[np.ndarray] = None,
            q_5inch: float = 0.0) -> "MuonTrack":
        """Fit the track to per-tube times *t* (ns) and charges *q* (p.e.) of inner-detector cables.

        *q17* is the total 17-inch charge for the charge-consistency term (default:
        sum of *q* over the 17-inch cables).  By default only the 17-inch tubes enter
        the time fit (the 20-inch tubes have a larger transit-time spread).
        *cable_od*, *t_od*: outer-detector hits (cables 1879-2119) whose time ordering
        enters the candidate score (orientation of the track).  *q_5inch*: total charge
        of the chimney tubes; above ``chimney_q_min`` the chimney seed and penalty are used."""
        chimney = bool(q_5inch is not None and q_5inch >= self.chimney_q_min)
        cable = np.asarray(cable); t = np.asarray(t, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
        P_od = None
        if cable_od is not None and t_od is not None:
            cable_od = np.asarray(cable_od); t_od = np.asarray(t_od, dtype=np.float64)
            m = (cable_od >= N_ID) & (cable_od < 2120) & np.isfinite(t_od)
            if m.sum() >= self.od_min_hits:
                P_od, t_od = self.xyz[cable_od[m]], t_od[m]
            else:
                t_od = None
        fin = np.isfinite(q)
        is17 = cable < N_ID17
        if q17 is None:
            q17 = float(q[is17 & fin].sum())
        qdense = None
        if self.charge_model is not None:
            qdense = np.zeros(N_ID17)
            np.add.at(qdense, cable[is17 & fin], q[is17 & fin])
        sel = (cable < (N_ID if use_20inch else N_ID17)) & np.isfinite(t) & (q >= self.q_min)
        cable, t, q = cable[sel], t[sel], q[sel]
        if len(t) < self.min_hits:
            e = np.array([0.0, 0.0, R_PMT]); x = -e
            return MuonTrack(e, x, np.nan, np.nan, False, 0, 0, len(t), np.nan, np.nan)
        P = self.xyz[cable]
        seeds = []
        if self.seed_mode in ("scan", "both"):
            seeds += [(e, x, t0, False) for e, x, t0 in self.scan(P, t, q, q17, qdense, P_od, t_od, chimney)]
        if self.seed_mode in ("cluster", "both"):
            seeds += [(e, x, None, self.try_flip) for e, x in self.seeds(P, t, q)]
        if chimney:
            # Kat's checkChimney: entrance at the charge centroid of the ID hits above z = 800 cm
            top = P[:, 2] > CHIMNEY_Z
            if top.sum() >= 3:
                w = np.sqrt(np.clip(q[top], 0, None)) + 1e-3
                e_ch = unit((P[top] * w[:, None]).sum(0) / w.sum()) * R_PMT
            else:
                e_ch = CHIMNEY_POINT.copy()
            for e, x, t0, _ in list(seeds)[:2]:
                if np.linalg.norm(x - e_ch) > 100.0:
                    seeds.append((e_ch, x, None, False))
        best = None
        for k, (e, x, t0, flip) in enumerate(seeds):
            trials = [(e, x)] + ([(x, e)] if flip else [])
            for j, (ee, xx) in enumerate(trials):
                tr = self._fit_from(P, t, q, ee, xx, t0, qdense)
                tr.seed = 2 * k + j
                tr.score = self.score(tr, q17, qdense, P_od, t_od, chimney)
                if best is None or tr.score > best.score:
                    best = tr
        # orientation check: the flipped track must be worse
        if self.try_flip and best is not None and np.isfinite(best.t0):
            tr = self._fit_from(P, t, q, best.exit.copy(), best.entrance.copy(), None, qdense)
            tr.seed = -1
            tr.score = self.score(tr, q17, qdense, P_od, t_od, chimney)
            if tr.score > best.score:
                best = tr
        best.q17 = q17
        best.chimney = chimney
        best.q_expected = float(self.expected_charge(best.impact))
        if self.charge_model is not None and qdense is not None and not np.isfinite(best.pattern_score):
            best.pattern_score, best.charge_scale = self.charge_model.score(qdense, best.entrance, best.exit)
        return best

    def residuals(self, track: "MuonTrack", cable: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Time residuals of the given tubes with respect to *track*."""
        P = self.xyz[np.asarray(cable)]
        return np.asarray(t, dtype=np.float64) - self.predict(P, track.entrance, track.exit, track.t0, track.n_eff)


# ---------------------------------------------------------------------------
# event-level convenience
# ---------------------------------------------------------------------------
def muon_row(header, ev_index: int, mh: np.ndarray, track: Optional[MuonTrack], pmts: PMTTable) -> np.ndarray:
    """Fill one MUON_DTYPE row from the header, the per-tube hits and the fitted track."""
    row = np.zeros(1, dtype=MUON_DTYPE)[0]
    row["index"] = ev_index
    row["event"] = header.event_number; row["run"] = header.run
    row["unix_time"] = header.time_s
    row["timestamp"] = getattr(header, "timestamp", 0); row["trigger"] = header.trigger_type & 0xFFFFFFFF
    idm = mh["cable"] < N_ID
    row["nhit"] = int(idm.sum()); row["nhit_od"] = int((~idm).sum()); row["nsat"] = int(mh["saturated"][idm].sum())
    row["n200_od"] = n200_od(mh["t"][~idm])
    row["q17"] = mh["q"][mh["cable"] < N_ID17].sum()
    row["q20"] = mh["q"][idm & (mh["cable"] >= N_ID17)].sum()
    od = mh[~idm]
    row["q_od"] = od["q"].sum()
    row["q_5inch"] = od["q"][(od["cable"] >= 2120) & (od["cable"] <= 2125)].sum()
    if track is not None and np.isfinite(track.t0):
        e, x, u = track.entrance, track.exit, track.direction
        row["ok"] = True
        row["converged"] = track.converged; row["n_iter"] = track.n_iter; row["n_used"] = track.n_used
        row["frac_used"] = track.frac_used; row["sigma_t"] = track.sigma_t; row["chi2"] = track.chi2
        row["ex"], row["ey"], row["ez"] = e; row["xx"], row["xy"], row["xz"] = x; row["ux"], row["uy"], row["uz"] = u
        row["t0"] = track.t0; row["cos_zenith"] = track.cos_zenith; row["azimuth"] = track.azimuth
        row["impact"] = track.impact; row["l_ls"] = track.l_ls; row["l_bo"] = track.l_bo; row["n_eff"] = track.n_eff
        row["seed"] = track.seed; row["score"] = track.score
        row["q_expected"] = track.q_expected
        row["pattern_score"] = track.pattern_score; row["charge_scale"] = track.charge_scale
        row["od_score"] = track.od_score; row["n_od_used"] = track.n_od_used
        row["chimney"] = track.chimney; row["chimney_dist"] = track.chimney_dist
        row["delta_q"] = row["q17"] - (DQDX_LS * track.l_ls + DQDX_BO * track.l_bo)   # residual charge (showers)
    return row
