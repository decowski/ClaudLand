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


def test_source_peak_nhit_prefers_highest_significant_peak():
    """68Ge-like spectrum: big background pile-up at ~150 hits, source peak at ~250 hits."""
    from claudland.zscan import source_peak_nhit, source_window
    rng = np.random.default_rng(3)
    nh = np.concatenate([rng.normal(150, 15, 2000), rng.normal(250, 15, 900), rng.integers(300, 600, 60)])
    peak = source_peak_nhit(nh)
    assert 240 < peak < 260
    lo, hi = source_window(nh)
    assert lo < 250 < hi
    # single-peak (60Co-like) spectrum with a small background continuum
    co = np.concatenate([rng.normal(620, 30, 3000), rng.integers(100, 500, 300)])
    assert 600 < source_peak_nhit(co) < 640


def test_run_header_source_position_parsing():
    from claudland.banks import RunHeader
    cases = {"+3.50 m": 350.0, "Ge68 in balloon center": 0.0, "Ge68 +6.00 m": 600.0, "Ge68 (-2.0 m)": -200.0,
             "Ge68 at +5.25": 525.0, "Ge68 at (-5.25)": -525.0, " Ge68 at 0.00 m": 0.0, "Co60 +1.50 m": 150.0,
             "+350 cm": 350.0, "normal run": None, "co 60 run": None, "co 60 run, 20' PMTs off": None,
             "Co-60 source - -3m": -300.0, "Co60 --5.25 m": -525.0, "source Co60 +5.25 m": 525.0}
    for comment, expected in cases.items():
        assert RunHeader(0, 0, 0, 0, "", "", comment).source_z_cm == expected, comment


def test_particle_energy_tables():
    import pytest
    from claudland.evis import ParticleEnergy, NaturalSpline
    from claudland import config
    if not ParticleEnergy.available():
        pytest.skip("private ParticleEnergy tables not available")
    pe = ParticleEnergy.load()
    g = pe.tables["gamma"]
    # the spline reproduces the table nodes and the inverse undoes the forward map
    assert np.allclose(pe.gamma_visible(g.e_real), g.e_real * g.ratio)
    assert np.allclose(pe.visible_to_gamma(pe.gamma_visible([0.7, 1.5, 3.0])), [0.7, 1.5, 3.0], rtol=2e-3)
    assert abs(pe.source_visible_energy("source-60Co") - 2.3425) < 0.005
    assert abs(pe.source_visible_energy("source-68Ge") - 0.8456) < 0.002
    assert pe.source_visible_energy("normal") is None
    # positron table at 1.022 MeV (no kinetic energy) == two 0.511 MeV gammas
    assert abs(pe.positron_visible(1.022) - pe.source_visible_energy("source-68Ge")) < 1e-3
    s = NaturalSpline([0, 1, 2, 3], [0, 1, 8, 27])
    assert abs(float(s(1.5)) - 3.375) < 0.6
