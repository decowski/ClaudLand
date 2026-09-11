import numpy as np

from claudland.geometry import PMTTable
from claudland.vertex import path_lengths, VertexFitter
from claudland.vertex_ml import MLVertexFitter
from claudland.zscan import TimePDF, event_time


def _synthetic_pdf(lo=-40.0, hi=220.0, dt=0.5, sigma=2.5, tau_fast=10.0, frac_fast=0.8, tau_slow=70.0):
    """Gaussian-resolved double exponential (Detwiler Fig. 4.3), same table for all classes."""
    t = np.arange(lo, hi, dt) + dt / 2
    x = np.linspace(-60, 400, 4000)
    emis = frac_fast * np.exp(-x / tau_fast) / tau_fast + (1 - frac_fast) * np.exp(-x / tau_slow) / tau_slow
    emis[x < 0] = 0
    kern = np.exp(-0.5 * (x - x.mean()) ** 2 / sigma ** 2)
    dens = np.convolve(emis, kern, mode="same")
    dens = np.interp(t, x, dens)
    dens = np.maximum(dens / (dens.sum() * dt), 1e-6 * dens.max())
    lp = np.log(dens)
    g = np.gradient(lp, dt)
    h = np.gradient(g, dt)
    shape = (2, 4, 4, len(t))
    return TimePDF(lo, dt, np.broadcast_to(lp, shape).copy(), np.broadcast_to(g, shape).copy(),
                   np.broadcast_to(h, shape).copy()), t, dens


def test_path_lengths():
    pm = PMTTable.load()
    P = pm.xyz[:1879]
    d_ls, d_bo, u, d = path_lengths(np.zeros(3), P)
    assert np.allclose(d_ls, 650.0, atol=0.5)
    assert np.allclose(d_ls + d_bo, d)
    d_ls, d_bo, u, d = path_lengths(np.array([0, 0, 600.0]), P)
    assert np.allclose(d_ls + d_bo, d) and d_ls.min() < 100 and d_bo.min() > 150


def test_timepdf_interpolation():
    pdf, t, dens = _synthetic_pdf()
    ti = np.zeros(3, int); qi = np.zeros(3, int); di = np.zeros(3, int)
    lp, g, h = pdf.evaluate(np.array([0.0, 20.0, 300.0]), ti, qi, di)
    assert abs(lp[0] - np.log(np.interp(0.0, t, dens))) < 0.05
    assert g[1] < 0 and np.isfinite(lp[2])     # falling tail; out-of-range clamps


def test_ml_fitter_recovers_vertex():
    rng = np.random.default_rng(3)
    pm = PMTTable.load()
    pdf, t, dens = _synthetic_pdf()
    v = 19.0
    true = np.array([100.0, -50.0, 300.0]); T0 = -280.0
    cables = rng.choice(1325, size=600, replace=False)
    P = pm.xyz[cables]
    tof = np.linalg.norm(P - true, axis=1) / v
    # draw residuals from the density
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    res = np.interp(rng.random(len(cables)), cdf, t)
    times = T0 + tof + res
    q = np.ones(len(cables))
    fit = MLVertexFitter(pm, pdf, v_ls=v, prefit=VertexFitter(pm, v_ls=v))
    r = fit.fit(cables, times, q)
    assert r.ok
    assert np.linalg.norm(r.xyz - true) < 25.0
    assert abs(r.t0 - T0) < 5.0


def test_event_time():
    tau = np.concatenate([np.random.default_rng(1).normal(-300, 2, 200), np.array([-250.0, -200.0, -100.0])])
    ev = np.zeros(len(tau), int)
    T = event_time(tau, ev, 1)
    assert abs(T[0] + 300) < 1.0


def test_charge_time_fitter_recovers_vertex_and_energy():
    from claudland.vertex_ml import ChargeTimeVertexFitter
    from claudland.energy import EnergyEstimator
    rng = np.random.default_rng(7)
    pm = PMTTable.load()
    pdf, t, dens = _synthetic_pdf()
    v = 19.0
    em = EnergyEstimator(pm)
    true = np.array([-80.0, 120.0, -250.0]); T0 = -290.0; E_true = 2.5
    # hit pattern from the light model, single p.e. charges
    mu = em.expected_pe_per_mev(true) * E_true
    nhit = rng.poisson(mu)
    cables = np.flatnonzero(nhit > 0)
    P = pm.xyz[cables]
    tof = np.linalg.norm(P - true, axis=1) / v
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    res = np.interp(rng.random(len(cables)), cdf, t)
    times = T0 + tof + res
    q = nhit[cables].astype(float)
    fit = ChargeTimeVertexFitter(pm, pdf, v_ls=v, energy_model=em, charge_model="hit", prefit=VertexFitter(pm, v_ls=v))
    r = fit.fit(cables, times, q)
    assert r.ok
    assert np.linalg.norm(r.xyz - true) < 25.0
    assert abs(r.energy - E_true) / E_true < 0.15
