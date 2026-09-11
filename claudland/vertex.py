"""Time-of-flight vertex fitter.

Implements the iterative "push" fitter used by KamLAND (Detwiler thesis
Sect. 4.3, ``Kat/src/KatLTVertex.cc``) as a weighted Gauss-Newton fit of the
event position ``r`` and time ``T`` to the hit times::

    t_i = T + |P_i - r| / v_eff + noise

Only hits inside a window around the peak of the time-of-flight-corrected
time distribution are used, which suppresses the scintillator's slow
emission tail, scattering/reflection and dark hits.  The window shrinks from
``window_start`` to ``window_final`` (ns) as the fit converges.

A charge-weighted centroid, radially expanded by ``1/0.62`` (Detwiler), is the
starting point.  The effective light speed defaults to Kat's 16.95 cm/ns; an
optional two-medium model (scintillator inside the balloon, buffer oil
outside) is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import PMTTable, BALLOON_RADIUS_CM, N_ID

__all__ = ["VertexFitter", "VertexResult", "path_lengths"]


def path_lengths(r: np.ndarray, P: np.ndarray, radius: float = BALLOON_RADIUS_CM):
    """Path lengths from vertex ``r`` to PMT positions ``P`` inside (``d_ls``) and
    outside (``d_bo``) the balloon sphere, plus unit vectors r->P and total distances."""
    r = np.asarray(r, dtype=float)
    dvec = P - r[None, :]
    d = np.linalg.norm(dvec, axis=1)
    u = dvec / np.maximum(d, 1e-6)[:, None]
    rr = float(np.dot(r, r))
    if rr >= radius ** 2:
        # vertex outside the scintillator: the path may still cross the balloon twice; ignore (all buffer oil)
        return np.zeros_like(d), d, u, d
    ur = u @ r
    disc = ur ** 2 - (rr - radius ** 2)
    s = -ur + np.sqrt(np.clip(disc, 0.0, None))
    d_ls = np.clip(s, 0.0, d)
    return d_ls, d - d_ls, u, d


@dataclass
class VertexResult:
    """Result of a window vertex fit: position, event time, residual rms and convergence flags."""
    x: float
    y: float
    z: float
    t0: float                 #: event time (ns, same reference as the hit times)
    n_used: int               #: hits inside the final time window
    n_hits: int               #: hits offered to the fitter
    sigma_t: float            #: rms of the time residuals of the used hits (ns)
    iterations: int
    converged: bool
    ok: bool
    prefit: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def r(self) -> float:
        """Distance of the vertex from the detector centre (cm)."""
        return float(np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2))

    @property
    def rho(self) -> float:
        """Cylindrical radius of the vertex (cm)."""
        return float(np.hypot(self.x, self.y))

    @property
    def xyz(self) -> np.ndarray:
        """Vertex position as a length-3 array (cm)."""
        return np.array([self.x, self.y, self.z])


class VertexFitter:
    """Windowed Gauss-Newton vertex fitter (Kat/Detwiler style) with a shrinking time window."""
    def __init__(self, pmts: PMTTable, v_ls: float = 16.95, v_bo: Optional[float] = None,
                 window_start: float = 40.0, window_final: float = 12.0,
                 max_iter: int = 40, max_step: float = 100.0, max_radius: float = 800.0,
                 min_charge: float = 0.3, prefit_expand: float = 1.0 / 0.62,
                 tol: float = 0.5, use_charge_weight: bool = False):
        self.pmts = pmts
        self.P = pmts.xyz
        self.v_ls = v_ls
        self.v_bo = v_bo
        self.window_start = window_start
        self.window_final = window_final
        self.max_iter = max_iter
        self.max_step = max_step
        self.max_radius = max_radius
        self.min_charge = min_charge
        self.prefit_expand = prefit_expand
        self.tol = tol
        self.use_charge_weight = use_charge_weight

    # -- light propagation -----------------------------------------------------
    def tof(self, r: np.ndarray, P: np.ndarray):
        """Time of flight (ns) from ``r`` to PMT positions ``P`` and the unit vectors r->P."""
        d_ls, d_bo, u, d = path_lengths(r, P)
        if self.v_bo is None:
            return d / self.v_ls, u, d
        return d_ls / self.v_ls + d_bo / self.v_bo, u, d

    # -- fit -----------------------------------------------------------------------
    def prefit(self, cable, q) -> np.ndarray:
        """Starting point: charge-weighted centroid of the hit tubes scaled by 1/0.62."""
        w = np.clip(q, 0, 20)
        if w.sum() <= 0:
            return np.zeros(3)
        c = (self.P[cable] * w[:, None]).sum(axis=0) / w.sum()
        c *= self.prefit_expand
        rad = np.linalg.norm(c)
        if rad > self.max_radius:
            c *= self.max_radius / rad
        return c

    def fit(self, cable: np.ndarray, t: np.ndarray, q: np.ndarray, start: Optional[np.ndarray] = None) -> VertexResult:
        """Fit hits given as arrays of cable numbers, times (ns) and charges (p.e.).

        The fit proceeds in stages of decreasing time window (``window_start``
        -> ``window_final``).  In each stage the hits within ``window`` of the
        peak of the ToF-corrected time distribution are selected and
        Gauss-Newton steps are taken until the position moves by less than
        ``tol`` cm (or ``max_iter`` total iterations are used up).
        """
        cable = np.asarray(cable); t = np.asarray(t, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
        sel = (cable < N_ID) & np.isfinite(t) & (q >= self.min_charge)
        cable, t, q = cable[sel], t[sel], q[sel]
        n = len(t)
        if n < 4:
            return VertexResult(np.nan, np.nan, np.nan, np.nan, 0, n, np.nan, 0, False, False)
        P = self.P[cable]
        r = self.prefit(cable, q) if start is None else np.array(start, dtype=float)
        pre = r.copy()
        weights_q = np.clip(q, 0.3, 4.0) if self.use_charge_weight else np.ones(n)
        # window schedule
        windows = [self.window_start]
        while windows[-1] > self.window_final * 1.05:
            windows.append(max(self.window_final, windows[-1] * 0.6))
        converged = False
        T = np.nan
        used = np.ones(n, dtype=bool)
        it = 0
        last_step = np.inf
        for stage, window in enumerate(windows):
            stage_iter = 0
            center = None
            while it < self.max_iter and stage_iter < 12:
                it += 1; stage_iter += 1
                tof, u, d = self.tof(r, P)
                tau = t - tof
                if center is None:
                    center = self._peak(tau)
                used = np.abs(tau - center) <= window
                if used.sum() < 4:
                    center = self._peak(tau)
                    used = np.abs(tau - center) <= window * 2
                    if used.sum() < 4:
                        break
                w = weights_q * used
                T = np.sum(w * tau) / np.sum(w)
                e = tau - T
                # Jacobian of the residual wrt (x, y, z, T): de/dr = u/v_eff, de/dT = -1
                v_eff = d / np.maximum(tof, 1e-6)
                J = np.empty((n, 4))
                J[:, :3] = u / v_eff[:, None]
                J[:, 3] = -1.0
                JW = J * w[:, None]
                A = JW.T @ J
                b = JW.T @ e
                try:
                    delta = -np.linalg.solve(A + 1e-9 * np.eye(4), b)   # Gauss-Newton step
                except np.linalg.LinAlgError:
                    break
                dr = delta[:3]
                step = np.linalg.norm(dr)
                if step > self.max_step:
                    dr *= self.max_step / step
                    step = self.max_step
                r = r + dr
                T = T + delta[3]
                center = T                      # follow the event time within the stage
                rad = np.linalg.norm(r)
                if rad > self.max_radius:
                    r *= self.max_radius / rad
                last_step = step
                if step < self.tol:
                    break
            if it >= self.max_iter:
                break
        converged = last_step < max(self.tol, 2.0)
        tof, u, d = self.tof(r, P)
        tau = t - tof
        if np.isfinite(T):
            used = np.abs(tau - T) <= windows[-1]
            e = tau - T
        n_used = int(used.sum())
        sigma = float(np.sqrt(np.mean(e[used] ** 2))) if (n_used > 1 and np.isfinite(T)) else np.nan
        ok = converged and n_used >= 4 and np.linalg.norm(r) < self.max_radius - 1e-3
        return VertexResult(float(r[0]), float(r[1]), float(r[2]), float(T), n_used, n, sigma, it, converged, ok, pre)

    @staticmethod
    def _peak(tau: np.ndarray) -> float:
        """Centre of the most populated 2 ns bin (3-bin smoothed) of the time distribution."""
        lo, hi = np.percentile(tau, [1, 99])
        edges = np.arange(lo - 4, hi + 6, 2.0)
        h, _ = np.histogram(tau, bins=edges)
        hs = np.convolve(h, [1, 1, 1], mode="same")
        k = int(hs.argmax())
        return 0.5 * (edges[k] + edges[k + 1])

    def residuals(self, res: VertexResult, cable, t):
        """Time-of-flight corrected times minus the event time for the given hits."""
        tof, u, d = self.tof(res.xyz, self.P[np.asarray(cable)])
        return np.asarray(t) - tof - res.t0
