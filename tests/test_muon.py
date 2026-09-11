"""Muon track fitter: geometry helpers, hit preparation and synthetic-track recovery."""
import numpy as np
import pytest

from claudland.geometry import PMTTable, N_ID17, N_ID
from claudland.reco import HIT_DTYPE
from claudland import muon as M


PMTS = PMTTable.load()


def _track(e_dir, x_dir):
    e = M.unit(np.asarray(e_dir, float)) * M.R_PMT
    x = M.unit(np.asarray(x_dir, float)) * M.R_PMT
    u = M.unit(x - e)
    return e, x, u, float(np.linalg.norm(x - e))


def test_chord_and_impact():
    e, x, u, L = _track([0, 0, 1], [0, 0, -1])          # through the centre
    assert M.impact_parameter(e, u) < 1e-6
    assert abs(M.chord_length(0.0, M.R_LS) - 2 * M.R_LS) < 1e-9
    assert M.chord_length(700.0, M.R_LS) == 0.0
    s = M.sphere_crossings(e, u, M.R_LS)
    assert s is not None and abs((s[1] - s[0]) - 2 * M.R_LS) < 1e-6
    assert M.sphere_crossings(np.array([0.0, 800.0, 0.0]), np.array([1.0, 0.0, 0.0]), M.R_LS) is None


def test_first_light_is_minimum_over_emission_points():
    pmts = PMTS
    """The closed-form first-light time equals the minimum over emission points along the track."""
    e, x, u, L = _track([1, 1, 1], [-1, 0.2, -1])
    n = 1.5
    P = pmts.xyz[:200]
    t = M.first_light_time(P, e, u, L, 0.0, n)
    s = np.linspace(0, L, 4001)
    em = e[None, :] + s[:, None] * u[None, :]
    d = np.linalg.norm(P[:, None, :] - em[None, :, :], axis=2)
    brute = (s[None, :] / M.C_CM_NS + n * d / M.C_CM_NS).min(axis=1)
    assert np.all(t <= brute + 1e-9)
    assert np.max(brute - t) < 0.02            # ns; limited by the grid step


def test_kat_index_table():
    assert M.kat_index(0) == 1.65 and M.kat_index(800) == 1.4
    assert 1.4 < M.kat_index(700) < 1.5


def test_muon_hits_picks_unsaturated_gain():
    h = np.zeros(5, dtype=HIT_DTYPE)
    h["cable"] = [7, 7, 7, 9, 2000]
    h["gain"] = [0, 1, 2, 0, 0]
    h["t"] = [-250.0, -249.0, -248.0, -240.0, -230.0]
    h["q"] = [400.0, 1500.0, 9000.0, 3.0, 5.0]
    h["saturated"] = [True, True, False, False, False]
    mh = M.muon_hits(h)
    c7 = mh[mh["cable"] == 7][0]
    assert np.isclose(c7["t"], -250.0)      # time from the high gain
    assert np.isclose(c7["q"], 9000.0)      # charge from the first unsaturated gain
    assert c7["gain"] == 2
    assert set(mh["cable"]) == {7, 9, 2000}


def test_is_muon():
    assert M.is_muon(20000, 0) and M.is_muon(600, 5) and not M.is_muon(600, 2) and not M.is_muon(300, 10)


def test_charge_model_normalisation():
    pmts = PMTS
    cm = M.MuonChargeModel(pmts)
    e, x, u, L = _track([0.2, 0.1, 1], [0, 0, -1])
    b = M.impact_parameter(e, u)
    mu = cm.expected(e, x)
    mip = M.DQDX_LS * M.chord_length(b, M.R_LS) + M.DQDX_BO * (M.chord_length(b, M.R_BO) - M.chord_length(b, M.R_LS))
    assert mu.shape == (N_ID17,)
    assert 0.8 < mu.sum() / mip < 1.3           # acceptance summed over tubes ≈ the standard yields
    # a buffer-oil track has no scintillation component
    e2, x2, u2, L2 = _track([1, 0, 0.3], [1, 0, -0.3])
    assert M.impact_parameter(e2, u2) > M.R_LS
    mu2 = cm.expected(e2, x2)
    assert mu2.sum() < 0.1 * mu.sum()


def _synthetic(pmts, e, x, n=1.5, seed=1):
    rng = np.random.default_rng(seed)
    u = M.unit(x - e); L = np.linalg.norm(x - e)
    P = pmts.xyz[:N_ID17]
    t = M.first_light_time(P, e, u, L, -250.0, n)
    t = t + rng.normal(0, 2.5, len(P)) + rng.exponential(8.0, len(P)) * (rng.random(len(P)) < 0.4)
    mu = M.MuonChargeModel(pmts).expected(e, x)
    q = rng.gamma(4.0, mu / 4.0 + 0.1) + rng.exponential(2.0, len(P))
    return t, q


@pytest.mark.parametrize("charge", [False, True])
def test_fit_recovers_ls_track(charge):
    pmts = PMTS
    e, x, u, L = _track([300, -200, 700], [-500, 100, -600])
    t, q = _synthetic(pmts, e, x)
    f = M.MuonTrackFitter(pmts, charge_model=M.MuonChargeModel(pmts) if charge else None, n_scan=100)
    tr = f.fit(np.arange(N_ID17), t, q)
    assert tr.converged
    assert np.linalg.norm(tr.entrance - e) < 40 and np.linalg.norm(tr.exit - x) < 40
    assert np.degrees(np.arccos(np.clip(tr.direction @ u, -1, 1))) < 2.0
    assert abs(tr.t0 + 250.0) < 4.0
    assert tr.l_ls > 1000 and tr.cos_zenith > 0.8


def test_fit_recovers_oil_clipper():
    pmts = PMTS
    e, x, u, L = _track([1, 0.2, 0.5], [1, -0.3, -0.4])
    assert M.impact_parameter(e, u) > M.R_LS
    t, q = _synthetic(pmts, e, x, seed=3)
    f = M.MuonTrackFitter(pmts, charge_model=M.MuonChargeModel(pmts), n_scan=100)
    tr = f.fit(np.arange(N_ID17), t, q)
    assert tr.converged and tr.l_ls == 0.0
    assert np.linalg.norm(tr.entrance - e) < 80 and np.linalg.norm(tr.exit - x) < 80
    assert tr.cos_zenith > 0                     # orientation (entrance above exit) recovered


def test_muon_row_fields():
    pmts = PMTS
    from claudland.banks import Header
    e, x, u, L = _track([0, 1, 1], [0, -1, -1])
    t, q = _synthetic(pmts, e, x)
    f = M.MuonTrackFitter(pmts, n_scan=60)
    tr = f.fit(np.arange(N_ID17), t, q)
    mh = np.zeros(N_ID17, dtype=M.MUHIT_DTYPE); mh["cable"] = np.arange(N_ID17); mh["t"] = t; mh["q"] = q
    hdr = Header(1, 0, 1467, 0, 42, 1_000_000_000, 0, 0, 0, 0x0a000002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    row = M.muon_row(hdr, 5, mh, tr, pmts)
    assert row["event"] == 42 and row["ok"] and row["nhit"] == N_ID17 and np.isclose(row["q17"], q.sum(), rtol=1e-5)
    assert abs(row["l_ls"] - tr.l_ls) < 1e-3 and np.isclose(row["delta_q"], row["q17"] - (M.DQDX_LS * tr.l_ls + M.DQDX_BO * tr.l_bo), rtol=1e-4)


def test_od_timing_fixes_orientation():
    """OD hit times growing along the track (slope 1/c) favour the right orientation."""
    pmts = PMTS
    e, x, u, L = _track([1, 0.2, 0.5], [1, -0.3, -0.4])         # buffer-oil clipper
    f = M.MuonTrackFitter(pmts, n_scan=60)
    rng = np.random.default_rng(5)
    od_cables = np.arange(N_ID, 2120)
    P_od = pmts.xyz[od_cables]
    z = (P_od - e) @ u
    t_od = -250.0 + 60.0 + z / M.C_CM_NS + rng.normal(0, 25.0, len(z))
    right = M.MuonTrack(e, x, -250.0, 1.5, True, 1, 500, 600, 3.0, 1.0)
    wrong = M.MuonTrack(x, e, -250.0 + L / M.C_CM_NS, 1.5, True, 1, 500, 600, 3.0, 1.0)
    s_right = f.od_term(right.entrance, right.exit, right.t0, P_od, t_od)
    s_wrong = f.od_term(wrong.entrance, wrong.exit, wrong.t0, P_od, t_od)
    assert s_right < s_wrong
    assert 0.5 < s_right < 1.5                                   # ≈ 1 for residuals of one sigma
    # vectorised version agrees
    v = f.od_terms(np.array([right.entrance, wrong.entrance]), np.array([right.exit, wrong.exit]),
                   np.array([right.t0, wrong.t0]), P_od, t_od)
    assert np.allclose(v, [s_right, s_wrong])
    # too few OD hits -> no term
    assert f.od_term(e, x, -250.0, P_od[:2], t_od[:2]) == 0.0


def test_chimney_check():
    """A chimney muon (Q_5inch >= 100) is flagged; tracks missing the top are penalised; the seed helps."""
    pmts = PMTS
    f = M.MuonTrackFitter(pmts, n_scan=60)
    e, x, u, L = _track([0.1, 0.05, 1], [0.3, -0.2, -1])          # enters through the top
    e2, x2, u2, L2 = _track([1, 0, 0], [-1, 0, 0])                 # horizontal through the centre
    assert f.chimney_term(e, x)[0] == 0.0 and f.chimney_term(e2, x2)[0] > 1.0
    t, q = _synthetic(pmts, e, x)
    tr = f.fit(np.arange(N_ID17), t, q, q_5inch=500.0)
    assert tr.chimney and tr.chimney_dist < 150 and tr.converged
    assert np.linalg.norm(tr.entrance - e) < 40
    tr2 = f.fit(np.arange(N_ID17), t, q, q_5inch=0.0)
    assert not tr2.chimney
    mh = np.zeros(N_ID17, dtype=M.MUHIT_DTYPE); mh["cable"] = np.arange(N_ID17); mh["t"] = t; mh["q"] = q
    from claudland.banks import Header
    hdr = Header(1, 0, 1467, 0, 1, 1_000_000_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    row = M.muon_row(hdr, 1, mh, tr, pmts)
    assert row["chimney"] and abs(row["chimney_dist"] - tr.chimney_dist) < 1e-3
