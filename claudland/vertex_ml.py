"""Maximum-likelihood vertex fitter using empirical hit-time densities.

Following the idea of M. Batygov's "V2" fitter (Batygov thesis, App. A) the
vertex ``r`` and event time ``T`` maximise::

    log L(r, T) = sum_i log phi_{k(i)}( t_i - T - tof_i(r) )

where ``phi_k`` is the *measured* density of the time residual for hits of
class ``k`` = (tube type, charge bin, distance bin), tabulated from
calibration-source data with the vertex fixed at the known source position
(:func:`claudland.zscan.build_time_pdf`).  Using the full asymmetric density
(sharp rise from the PMT transit-time spread, long tail from scintillator
re-emission and scattering) instead of a hard time window and Gaussian errors
uses all hits with their proper weight; the charge dependence accounts for the
earlier arrival of the first of several photo-electrons.

The time of flight uses separate effective light speeds in the scintillator
(inside the 6.5 m balloon) and in the buffer oil.  Maximisation is a damped
Newton iteration with the analytic gradient of ``log phi`` and a numerical
Jacobian of the time of flight, started from the fast window fitter
(:class:`claudland.vertex.VertexFitter`) or from a charge centroid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import PMTTable, N_ID
from .vertex import VertexFitter, VertexResult, path_lengths
from .zscan import TimePDF

__all__ = ["MLVertexFitter", "MLVertexResult", "ChargeTimeVertexFitter", "ChargeTimeResult"]


@dataclass
class MLVertexResult(VertexResult):
    """Result of the maximum-likelihood time fit: position, event time, log-likelihood and convergence."""
    loglike: float = np.nan
    err: np.ndarray = None      #: (4,) parameter uncertainties from the inverse Hessian (x, y, z, T)


class MLVertexFitter:
    """Maximum-likelihood vertex fitter with empirical, charge-binned time PDFs (damped Newton)."""
    def __init__(self, pmts: PMTTable, pdf: TimePDF, v_ls: float, v_bo: Optional[float] = None,
                 prefit: Optional[VertexFitter] = None, max_iter: int = 40, tol: float = 1.0,
                 max_step: float = 100.0, max_radius: float = 800.0, min_charge: float = 0.3,
                 eps: float = 1.0, dark_fraction: float = 0.0):
        self.pmts = pmts
        self.P = pmts.xyz
        self.pdf = pdf
        self.v_ls = v_ls
        self.v_bo = v_bo if v_bo is not None else v_ls
        self.prefit = prefit or VertexFitter(pmts, v_ls=v_ls, v_bo=self.v_bo)
        self.max_iter = max_iter
        self.tol = tol
        self.max_step = max_step
        self.max_radius = max_radius
        self.min_charge = min_charge
        self.eps = eps
        self.dark_fraction = dark_fraction
        self.window_final = 12.0    #: |tau| window used to flag "used" hits downstream

    def tof(self, r, P):
        """Time of flight from the vertex to every given tube (ns), through scintillator and buffer oil."""
        d_ls, d_bo, u, d = path_lengths(r, P)
        return d_ls / self.v_ls + d_bo / self.v_bo, d

    def _eval(self, r, T, P, t, ti, qi):
        """log L, gradient (4,), Hessian (4,4), per-hit residuals."""
        tof, d = self.tof(r, P)
        di = np.clip(np.searchsorted(self.pdf.d_edges, d, side="right") - 1, 0, len(self.pdf.d_edges) - 2)
        tau = t - T - tof
        lp, g, h = self.pdf.evaluate(tau, ti, qi, di)
        # numerical Jacobian of tau wrt (x, y, z): d tau / d r_k = -(tof(r + e) - tof(r - e)) / 2e
        J = np.empty((len(t), 4))
        for k in range(3):
            e = np.zeros(3); e[k] = self.eps
            tp, _ = self.tof(r + e, P)
            tm, _ = self.tof(r - e, P)
            J[:, k] = -(tp - tm) / (2 * self.eps)
        J[:, 3] = -1.0
        grad = J.T @ g
        H = (J * h[:, None]).T @ J
        return float(lp.sum()), grad, H, tau

    def fit(self, cable, t, q, start: Optional[np.ndarray] = None, T_start: Optional[float] = None) -> MLVertexResult:
        """Fit the vertex (and energy) to the given tubes; returns the result object of the fitter."""
        cable = np.asarray(cable); t = np.asarray(t, dtype=float); q = np.asarray(q, dtype=float)
        sel = (cable < N_ID) & np.isfinite(t) & (q >= self.min_charge)
        cable, t, q = cable[sel], t[sel], q[sel]
        n = len(t)
        if n < 5:
            return MLVertexResult(np.nan, np.nan, np.nan, np.nan, 0, n, np.nan, 0, False, False)
        P = self.P[cable]
        ti, qi, _ = self.pdf.bins(cable, q, np.full(n, 850.0))
        # starting point
        if start is None:
            pre = self.prefit.fit(cable, t, q)
            if np.isfinite(pre.x) and pre.n_used >= 4:
                r = pre.xyz.copy(); T = pre.t0
            else:
                r = self.prefit.prefit(cable, q); T = np.nan
        else:
            r = np.array(start, dtype=float); T = T_start if T_start is not None else np.nan
        if not np.isfinite(T):
            tof, d = self.tof(r, P)
            T = float(np.median(t - tof))
        L, grad, H, tau = self._eval(r, T, P, t, ti, qi)
        lam = 0.0
        it = 0
        converged = False
        for it in range(1, self.max_iter + 1):
            # damped Newton: solve (H - lam I) delta = -grad  (H negative definite at a maximum)
            accepted = False
            for attempt in range(10):
                A = H - lam * np.eye(4)
                evals = np.linalg.eigvalsh(A)
                if evals.max() >= 0:      # not negative definite -> more damping
                    lam = max(lam * 4, abs(evals.max()) * 1.5 + 1e-3)
                    continue
                delta = -np.linalg.solve(A, grad)
                step = np.linalg.norm(delta[:3])
                if step > self.max_step:
                    delta *= self.max_step / step
                    step = self.max_step
                r_new = r + delta[:3]
                rad = np.linalg.norm(r_new)
                if rad > self.max_radius:
                    r_new *= self.max_radius / rad
                T_new = T + delta[3]
                L_new, grad_new, H_new, tau_new = self._eval(r_new, T_new, P, t, ti, qi)
                if L_new >= L:
                    r, T, L, grad, H, tau = r_new, T_new, L_new, grad_new, H_new, tau_new
                    lam *= 0.3
                    accepted = True
                    break
                if step < self.tol:
                    # a step below the tolerance does not improve L: we are at the maximum
                    # within the statistical flatness of the likelihood surface
                    accepted = True
                    break
                lam = lam * 4 if lam > 0 else 1e-2
            if not accepted:
                break
            if step < self.tol:
                converged = True
                break
        # uncertainties from the observed information
        err = np.full(4, np.nan)
        try:
            cov = np.linalg.inv(-H)
            if np.all(np.diag(cov) > 0):
                err = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            pass
        used = np.abs(tau) <= 12.0
        sigma = float(np.sqrt(np.mean(tau[used] ** 2))) if used.sum() > 1 else np.nan
        ok = converged and np.linalg.norm(r) < self.max_radius - 1e-3
        return MLVertexResult(float(r[0]), float(r[1]), float(r[2]), float(T), int(used.sum()), n, sigma, it,
                              converged, ok, loglike=L, err=err)

    def residuals(self, res: VertexResult, cable, t):
        """Time-of-flight-corrected hit times minus the event time."""
        tof, d = self.tof(res.xyz, self.P[np.asarray(cable)])
        return np.asarray(t) - tof - res.t0


# ---------------------------------------------------------------------------
# charge + time likelihood
# ---------------------------------------------------------------------------
from math import lgamma as _lgamma  # noqa: E402

_lgamma_vec = np.vectorize(_lgamma, otypes=[float])


@dataclass
class ChargeTimeResult(MLVertexResult):
    """Result of the joint charge + time fit, with the fitted energy."""
    energy: float = np.nan      #: fitted visible energy (MeV)
    energy_err: float = np.nan
    loglike_time: float = np.nan
    loglike_charge: float = np.nan


class ChargeTimeVertexFitter(MLVertexFitter):
    """Joint likelihood of hit times and of the charge / hit pattern.

    Parameters are (x, y, z, T, ln E).  The time term is that of
    :class:`MLVertexFitter`; the charge term uses the expected number of
    photo-electrons per tube ``mu_i = E * eta_i * f_i(r) + dark_i`` from the
    light-collection model of :class:`claudland.energy.EnergyEstimator`:

    ``charge_model="hit"``      Bernoulli on the hit pattern (Detwiler Eq. 4.6):
                                 hit -> log(1 - exp(-mu)), no hit -> -mu
    ``charge_model="poisson"``  continuous Poisson on the measured charge q (p.e.)
                                 for hit tubes, -mu for tubes without a hit

    ``charge_weight`` scales the charge term (1 = formal likelihood).  The
    energy is fitted simultaneously, so the result carries ``energy``.
    """

    def __init__(self, pmts, pdf, v_ls, v_bo=None, energy_model=None, charge_model: str = "hit",
                 charge_weight: float = 1.0, **kw):
        super().__init__(pmts, pdf, v_ls, v_bo, **kw)
        if energy_model is None:
            from .energy import EnergyEstimator
            energy_model = EnergyEstimator(pmts)
        self.energy_model = energy_model
        self.charge_model = charge_model
        self.charge_weight = charge_weight

    # -- charge term ------------------------------------------------------------------
    def _alpha(self, r):
        """eta_i f_i(r) for the live tubes (0 elsewhere); charge yield for the Poisson model."""
        em = self.energy_model
        eta = em.eta_charge if self.charge_model == "poisson" else em.eta
        return np.where(em.live, eta * em.f(r), 0.0)

    def _charge_terms(self, r, E, hit, qhit):
        """log L_q, d/d(x,y,z,lnE), Hessian (4x4 in x,y,z,lnE)."""
        em = self.energy_model
        live = em.live
        alpha = self._alpha(r)
        dalpha = np.empty((len(alpha), 3))
        for k in range(3):
            e = np.zeros(3); e[k] = self.eps
            dalpha[:, k] = (self._alpha(r + e) - self._alpha(r - e)) / (2 * self.eps)
        mu = E * alpha + em.dark
        mu = np.maximum(mu, 1e-9)
        # derivatives of mu wrt (x, y, z, lnE)
        J = np.empty((len(alpha), 4))
        J[:, :3] = E * dalpha
        J[:, 3] = E * alpha
        if self.charge_model == "poisson":
            q = qhit
            l = np.where(hit, -mu + q * np.log(mu) - _lgamma_vec(q + 1.0), -mu)
            dl = np.where(hit, -1.0 + q / mu, -1.0)
            d2l = np.where(hit, -q / mu ** 2, 0.0)
        else:
            em_ = np.exp(-mu)
            one = np.maximum(1.0 - em_, 1e-300)
            l = np.where(hit, np.log(one), -mu)
            dl = np.where(hit, em_ / one, -1.0)
            d2l = np.where(hit, -em_ / one ** 2, 0.0)
        l = np.where(live, l, 0.0); dl = np.where(live, dl, 0.0); d2l = np.where(live, d2l, 0.0)
        grad = J.T @ dl
        H = (J * d2l[:, None]).T @ J
        return float(l.sum()), grad, H

    def _eval5(self, theta, P, t, ti, qi, hit, qhit):
        r = theta[:3]; T = theta[3]; E = np.exp(theta[4])
        Lt, gt, Ht, tau = self._eval(r, T, P, t, ti, qi)
        Lq, gq, Hq = self._charge_terms(r, E, hit, qhit)
        w = self.charge_weight
        grad = np.zeros(5); H = np.zeros((5, 5))
        grad[:4] += gt; H[:4, :4] += Ht
        idx = [0, 1, 2, 4]
        grad[idx] += w * gq
        H[np.ix_(idx, idx)] += w * Hq
        return Lt + w * Lq, grad, H, tau, Lt, Lq

    def fit(self, cable, t, q, start=None, T_start=None, E_start=None) -> ChargeTimeResult:  # type: ignore[override]
        """Fit the vertex (and energy) to the given tubes; returns the result object of the fitter."""
        cable = np.asarray(cable); t = np.asarray(t, dtype=float); q = np.asarray(q, dtype=float)
        sel = (cable < N_ID) & np.isfinite(t) & (q >= self.min_charge)
        cable, t, q = cable[sel], t[sel], q[sel]
        n = len(t)
        if n < 5:
            return ChargeTimeResult(np.nan, np.nan, np.nan, np.nan, 0, n, np.nan, 0, False, False)
        P = self.P[cable]
        ti, qi, _ = self.pdf.bins(cable, q, np.full(n, 850.0))
        # starting point: time-only fit
        pre = super().fit(cable, t, q, start=start, T_start=T_start)
        if np.isfinite(pre.x):
            r = pre.xyz.copy(); T = pre.t0
        else:
            r = self.prefit.prefit(cable, q); T = float(np.median(t - self.tof(r, P)[0]))
        # hit pattern: tubes with a hit inside the energy model's signal window (same
        # definition as the light-yield calibration), evaluated at the starting vertex
        tau0 = t - T - self.tof(r, P)[0]
        win = self.energy_model.window
        inwin = (tau0 >= win[0]) & (tau0 <= win[1])
        hit = np.zeros(N_ID, dtype=bool); hit[cable[inwin]] = True
        qhit = np.zeros(N_ID); qhit[cable[inwin]] = q[inwin]
        if E_start is None:
            alpha = self._alpha(r)
            E_start = max((q.sum() if self.charge_model == "poisson" else -np.log(1 - min(hit[self.energy_model.live].mean(), 0.99)) * self.energy_model.live.sum()) / max(alpha.sum(), 1e-9), 0.05)
        theta = np.array([r[0], r[1], r[2], T, np.log(E_start)])
        L, grad, H, tau, Lt, Lq = self._eval5(theta, P, t, ti, qi, hit, qhit)
        lam = 0.0
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):
            accepted = False
            for attempt in range(10):
                A = H - lam * np.eye(5)
                evals = np.linalg.eigvalsh(A)
                if evals.max() >= 0:
                    lam = max(lam * 4, abs(evals.max()) * 1.5 + 1e-3)
                    continue
                delta = -np.linalg.solve(A, grad)
                step = np.linalg.norm(delta[:3])
                if step > self.max_step:
                    delta *= self.max_step / step
                    step = self.max_step
                delta[4] = np.clip(delta[4], -0.5, 0.5)
                th_new = theta + delta
                rad = np.linalg.norm(th_new[:3])
                if rad > self.max_radius:
                    th_new[:3] *= self.max_radius / rad
                L_new, g_new, H_new, tau_new, Lt_new, Lq_new = self._eval5(th_new, P, t, ti, qi, hit, qhit)
                if L_new >= L:
                    theta, L, grad, H, tau, Lt, Lq = th_new, L_new, g_new, H_new, tau_new, Lt_new, Lq_new
                    lam *= 0.3
                    accepted = True
                    break
                if step < self.tol and abs(delta[4]) < 0.01:
                    accepted = True
                    break
                lam = lam * 4 if lam > 0 else 1e-2
            if not accepted:
                break
            if step < self.tol and abs(delta[4]) < 0.01:
                converged = True
                break
        err = np.full(5, np.nan)
        try:
            cov = np.linalg.inv(-H)
            if np.all(np.diag(cov) > 0):
                err = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            pass
        used = np.abs(tau) <= 12.0
        sigma = float(np.sqrt(np.mean(tau[used] ** 2))) if used.sum() > 1 else np.nan
        r = theta[:3]
        ok = converged and np.linalg.norm(r) < self.max_radius - 1e-3
        E = float(np.exp(theta[4]))
        return ChargeTimeResult(float(r[0]), float(r[1]), float(r[2]), float(theta[3]), int(used.sum()), n, sigma, it,
                                converged, ok, loglike=L, err=err[:4], energy=E, energy_err=E * err[4],
                                loglike_time=Lt, loglike_charge=Lq)
