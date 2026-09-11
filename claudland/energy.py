"""Visible-energy estimators.

The light seen by PMT *i* for an event of energy *E* at position *r* is
modelled (Detwiler thesis Sect. 4.4) as::

    mu_i(E, r) = eta_i * f_i(r) * E + delta_i           [photoelectrons]
    f_i(r)     = Omega_i(r) exp(-d_i/Lambda) / (Omega_i(0) exp(-R/Lambda))
    Omega_i(r) = A_i (0.1 + 0.9 cos theta_i) / d_i^2

where ``eta_i`` is the light yield per MeV for a central event (0.220 p.e./MeV
for 17-inch, 0.293 p.e./MeV for 20-inch tubes -- so that a 60Co event at the
centre gives 2.506 MeV), ``delta_i`` the dark hits per tube per event,
``Lambda`` the absorption length (25 m) and ``theta_i`` the angle between the
PMT axis and the direction to the event.

Two estimators are provided:

* :meth:`EnergyEstimator.charge_energy` -- total (dark-subtracted) charge in
  the event time window divided by the expected light collection at ``r``.
* :meth:`EnergyEstimator.hit_energy` -- maximum-likelihood energy from the
  hit / no-hit pattern of the live tubes (Detwiler Eq. 4.6), which is
  insensitive to the charge calibration for events where most tubes see
  <~ 1 p.e.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import PMTTable, N_ID, N_ID17, PMT_RADIUS_CM

__all__ = ["EnergyEstimator", "EnergyResult", "CO60_ENERGY_MEV"]

CO60_ENERGY_MEV = 2.506   # summed gamma energy of 60Co (1.173 + 1.333 MeV)


@dataclass
class EnergyResult:
    """Energy estimates of one event from the charge (``e_charge``) and from the hit pattern (``e_hit``)."""
    e_charge: float      #: charge-based visible energy (MeV)
    e_hit: float         #: hit-pattern likelihood energy (MeV)
    q_window: float      #: charge in the signal window (p.e.), dark subtracted
    q_total: float       #: total ID charge (p.e.)
    q17: float
    q20: float
    n_window: int        #: hits in the signal window
    dark_hits: float     #: estimated dark hits in the window
    f_collect: float     #: relative light collection at the vertex (1 at centre)


class EnergyEstimator:
    """Visible-energy estimator: expected light per tube for a point source and the two estimators of Detwiler Ch. 4."""
    def __init__(self, pmts: PMTTable, eta17: float = 0.220, eta20: float = 0.293,
                 attenuation_cm: float = 2500.0, live: Optional[np.ndarray] = None,
                 dark_per_tube: Optional[np.ndarray] = None, eta: Optional[np.ndarray] = None,
                 eta_charge: Optional[np.ndarray] = None,
                 window: tuple = (-15.0, 85.0), dark_window: tuple = (-200.0, -40.0)):
        self.pmts = pmts
        self.P = pmts.xyz[:N_ID]
        self.normal = pmts.normal[:N_ID]
        self.area = pmts.area[:N_ID]
        # per-tube light yield at the centre (p.e./MeV); a calibrated array overrides the type averages
        self.eta = np.where(np.arange(N_ID) < N_ID17, eta17, eta20) if eta is None else np.asarray(eta, float)[:N_ID].copy()
        # yield used for the charge estimator (mean p.e. per MeV rather than hit probability)
        self.eta_charge = self.eta if eta_charge is None else np.asarray(eta_charge, float)[:N_ID].copy()
        self.lam = attenuation_cm
        self.live = np.ones(N_ID, dtype=bool) if live is None else np.asarray(live, dtype=bool)[:N_ID]
        self.dark = np.zeros(N_ID) if dark_per_tube is None else np.asarray(dark_per_tube, dtype=float)[:N_ID]
        self.window = window
        self.dark_window = dark_window
        self.scale = 1.0     #: extra multiplicative energy scale (from calibration)
        self._f0 = self._omega_exp(np.zeros(3))

    # -- light collection model --------------------------------------------------
    def _omega_exp(self, r: np.ndarray) -> np.ndarray:
        dvec = self.P - r[None, :]
        d = np.linalg.norm(dvec, axis=1)
        cos = np.einsum("ij,ij->i", -dvec / d[:, None], self.normal)   # angle at the PMT
        cos = np.clip(cos, 0.0, 1.0)
        omega = self.area * (0.1 + 0.9 * cos) / d ** 2
        return omega * np.exp(-d / self.lam)

    def f(self, r) -> np.ndarray:
        """Relative light collection ``f_i(r)`` for all ID tubes (1 at the centre)."""
        return self._omega_exp(np.asarray(r, dtype=float)) / self._f0

    def expected_pe_per_mev(self, r) -> np.ndarray:
        """``eta_i f_i(r)`` for live tubes (0 for dead ones)."""
        return np.where(self.live, self.eta * self.f(r), 0.0)

    # -- estimators ----------------------------------------------------------------
    def estimate(self, cable, tau, q, r) -> EnergyResult:
        """Energy from hits (cable numbers, ToF-corrected times ``tau`` relative to
        the event time in ns, charges in p.e.) at vertex ``r`` (cm)."""
        cable = np.asarray(cable); tau = np.asarray(tau, dtype=float); q = np.asarray(q, dtype=float)
        idm = (cable < N_ID) & np.isfinite(tau)
        cable, tau, q = cable[idm], tau[idm], q[idm]
        inwin = (tau >= self.window[0]) & (tau <= self.window[1])
        indark = (tau >= self.dark_window[0]) & (tau <= self.dark_window[1])
        wlen = self.window[1] - self.window[0]
        dlen = self.dark_window[1] - self.dark_window[0]
        dark_hits = indark.sum() * wlen / dlen if dlen > 0 else 0.0
        dark_q = q[indark].sum() * wlen / dlen if dlen > 0 else 0.0
        q_win = q[inwin].sum() - dark_q
        q17 = q[(cable < N_ID17)].sum()
        q20 = q[(cable >= N_ID17)].sum()
        alpha = self.expected_pe_per_mev(r)
        f_collect = alpha.sum() / np.where(self.live, self.eta, 0.0).sum()
        alpha_q = np.where(self.live, self.eta_charge * self.f(r), 0.0)
        e_charge = self.scale * q_win / alpha_q.sum() if alpha_q.sum() > 0 else np.nan
        # hit-pattern likelihood
        hit = np.zeros(N_ID, dtype=bool)
        hit[cable[inwin]] = True
        e_hit = self.scale * self._hit_likelihood_energy(hit, alpha, dark_hits)
        return EnergyResult(float(e_charge), float(e_hit), float(q_win), float(q.sum()), float(q17), float(q20),
                            int(inwin.sum()), float(dark_hits), float(f_collect))

    def _hit_likelihood_energy(self, hit: np.ndarray, alpha: np.ndarray, dark_hits: float) -> float:
        live = self.live & (alpha > 0)
        if live.sum() == 0:
            return np.nan
        a = alpha[live]
        h = hit[live]
        nlive = live.sum()
        delta = self.dark[live] if self.dark.any() else np.full(nlive, dark_hits / max(nlive, 1))
        # dL/dE = sum_hit a e^{-mu}/(1-e^{-mu}) - sum_nohit a ; monotonically decreasing in E
        def dl(E):
            """Distance from the vertex to every tube (cm)."""
            mu = a * E + delta
            em = np.exp(-mu)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(h, a * em / np.maximum(1 - em, 1e-300), -a)
            return term.sum()
        if h.all():
            return np.inf
        lo, hi = 0.0, 1.0
        while dl(hi) > 0 and hi < 1e4:
            hi *= 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if dl(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-4:
                break
        return 0.5 * (lo + hi)
