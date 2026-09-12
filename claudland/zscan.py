"""Calibration from the ⁶⁰Co z-scan hit caches (see ``scripts/extract_hits.py``).

All routines work on cached hits (samples / ADC sums) plus the known source
position of each run, so nothing here depends on a vertex fit:

* :func:`calibrate_center` -- per-channel time offsets (T0) and 1 p.e. charges
  (Q0) from the centre run (source at the origin: every tube is at the same
  distance, so the T0s do not depend on the assumed light speed).
* :func:`fit_velocities` -- effective light speeds in scintillator and buffer
  oil from the peak of the hit-time residual versus path length in each medium.
* :class:`TimePDF` / :func:`build_time_pdf` -- empirical probability densities
  of the time residual, binned in tube type, charge and distance, used by the
  maximum-likelihood fitter in :mod:`claudland.vertex_ml`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calib import TQCalibration, LAUNCH_OFFSET_NS
from .geometry import PMTTable, N_ID, N_ID17, BALLOON_RADIUS_CM
from .vertex import path_lengths

__all__ = ["load_cache", "hit_times", "hit_charges", "calibrate_center", "calibrate_light_yield",
           "fit_velocities", "TimePDF", "build_time_pdf", "event_time", "residuals_fixed_vertex",
           "source_peak_nhit", "source_window", "select_source_events", "filter_cache", "peak_fit"]

Q_EDGES = np.array([0.3, 1.5, 3.0, 6.0, np.inf])        # p.e. bins of the time PDF
D_EDGES = np.array([0.0, 400.0, 650.0, 900.0, np.inf])  # cm bins of the time PDF


def load_cache(path: str) -> Dict[str, np.ndarray]:
    """Load a hit cache written by ``scripts/extract_hits.py`` (events, hits, bin widths, ...)."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def source_peak_nhit(nhit: np.ndarray, bin_width: int = 10, min_frac: float = 0.1, smooth: int = 3) -> float:
    """Position of the calibration-source peak in a hit-multiplicity spectrum.

    The spectrum of a source run has the source peak plus, for a low-energy
    source such as 68Ge (1.022 MeV, ~250 hits on the 17-inch tubes), a
    background population piling up just above the trigger threshold that can
    contain more events than the source peak.  The source is therefore taken
    as the local maximum with the *largest* nhit among those reaching at least
    ``min_frac`` of the highest bin (after a ``smooth``-bin running average);
    for a 60Co run this is simply the main peak.
    """
    nhit = np.asarray(nhit)
    nhit = nhit[nhit > 0]
    if len(nhit) == 0:
        return float("nan")
    edges = np.arange(0, nhit.max() + 2 * bin_width, bin_width)
    counts, _ = np.histogram(nhit, bins=edges)
    hs = np.convolve(counts, np.ones(smooth) / smooth, mode="same")
    thresh = min_frac * hs.max()
    peaks = [k for k in range(1, len(hs) - 1) if hs[k] >= thresh and hs[k] >= hs[k - 1] and hs[k] > hs[k + 1]]
    if not peaks:
        peaks = [int(hs.argmax())]
    k = max(peaks)
    if 0 < k < len(hs) - 1:                        # parabolic refinement
        y0, y1, y2 = hs[k - 1], hs[k], hs[k + 1]
        den = y0 - 2 * y1 + y2
        dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        dk = 0.0
    return float(edges[k] + bin_width * (0.5 + dk))


def source_window(nhit: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> Tuple[int, int]:
    """``(nhit_min, nhit_max)`` window around the source peak (see :func:`source_peak_nhit`)."""
    peak = source_peak_nhit(nhit)
    return int(lo * peak), int(hi * peak)


def filter_cache(cache: Dict, keep: np.ndarray) -> Dict:
    """A shallow copy of *cache* restricted to the events with ``keep[ev] == True``
    (the ``ev`` indices of the hits are unchanged, so ``len(events)`` stays the same)."""
    out = dict(cache)
    out["hits"] = cache["hits"][keep[cache["hits"]["ev"]]]
    out["events"] = cache["events"].copy()
    out["events"]["nhit"] = np.where(keep, cache["events"]["nhit"], 0)   # excluded events drop out of nhit windows
    return out


def select_source_events(cache: Dict, calib: TQCalibration, pmts: PMTTable, source_xyz: Sequence[float],
                         v_ls: float = 17.6, nhit_window: Optional[Tuple[int, int]] = None,
                         max_dist: float = 150.0, min_nhit_fit: int = 20) -> Dict[str, np.ndarray]:
    """Boolean mask of the source events of a run: hit multiplicity inside
    ``nhit_window`` (default: :func:`source_window` of the run) *and* window-fitter
    vertex within ``max_dist`` cm of the known source position.

    For a low-energy source (68Ge, 1 MeV) the multiplicity window alone keeps
    20-40 % ambient background, which is spread over the whole detector and
    would smear every constant derived with the vertex fixed at the source;
    the position cut removes it (a 150 cm sphere is 1 % of the balloon volume)
    while the vertex resolution of ~30 cm keeps essentially all source events.
    Returns ``{"keep", "in_window", "fitted", "dist"}``."""
    from .vertex import VertexFitter
    nh = cache["events"]["nhit"]
    lo, hi = nhit_window if nhit_window is not None else source_window(nh)
    in_window = (nh >= lo) & (nh <= hi)
    h = cache["hits"]
    base = h["primary"] & (h["cable"] < N_ID) & np.isfinite(h["t_cfd"]) & in_window[h["ev"]]
    t = hit_times(cache, calib); q = hit_charges(cache, calib)
    fitter = VertexFitter(pmts, v_ls=v_ls)
    src = np.asarray(source_xyz, dtype=float)
    dist = np.full(len(nh), np.nan)
    order, ks, b = _group_bounds(h["ev"][base])
    idx_all = np.flatnonzero(base)[order]
    for a, c in zip(b[:-1], b[1:]):
        if c - a < min_nhit_fit:
            continue
        idx = idx_all[a:c]
        r = fitter.fit(h["cable"][idx], t[idx], q[idx])
        if r.ok:
            dist[ks[a]] = np.linalg.norm(r.xyz - src)
    fitted = np.isfinite(dist)
    keep = in_window & fitted & (dist <= max_dist)
    return {"keep": keep, "in_window": in_window, "fitted": fitted, "dist": dist}


def peak_fit(x: np.ndarray, bin_width: float = 10.0, half_range: float = 300.0, sigma0: float = 30.0,
             n_iter: int = 30) -> Dict[str, float]:
    """Gaussian + flat background fit to the distribution of *x* (e.g. reconstructed z of
    the source events).  Least squares on a histogram around the smoothed mode;
    returns ``mu, sigma, n_peak, bkg_per_bin, bkg_frac`` (background fraction
    within ±3 sigma) and ``n`` (entries used).  Robust against the ambient
    background under a low-energy source peak, which biases medians and
    inflates MAD-based widths."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return {"mu": np.nan, "sigma": np.nan, "n_peak": 0.0, "bkg_per_bin": np.nan, "bkg_frac": np.nan, "n": len(x)}
    edges = np.arange(x.min() - bin_width, x.max() + 2 * bin_width, bin_width)
    cnt, _ = np.histogram(x, bins=edges)
    sm = np.convolve(cnt, np.ones(3) / 3, mode="same")
    mode = 0.5 * (edges[sm.argmax()] + edges[sm.argmax() + 1])
    sel = np.abs(0.5 * (edges[:-1] + edges[1:]) - mode) <= half_range
    c = 0.5 * (edges[:-1] + edges[1:])[sel]; y = cnt[sel].astype(float)
    # parameters: amplitude A, mu, sigma, background B
    B = float(np.median(y[np.abs(c - mode) > 3 * sigma0])) if (np.abs(c - mode) > 3 * sigma0).sum() >= 3 else 0.0
    p = np.array([max(y.max() - B, 1.0), mode, sigma0, B])
    w = 1.0 / np.sqrt(np.maximum(y, 1.0))
    for _ in range(n_iter):
        A, mu, sg, B = p
        g = np.exp(-0.5 * ((c - mu) / sg) ** 2)
        f = A * g + B
        J = np.column_stack([g, A * g * (c - mu) / sg ** 2, A * g * (c - mu) ** 2 / sg ** 3, np.ones_like(c)])
        r = (y - f) * w
        step, *_ = np.linalg.lstsq(J * w[:, None], r, rcond=None)
        p_new = p + step
        p_new[2] = np.clip(abs(p_new[2]), 0.3 * bin_width, half_range)
        p_new[3] = max(p_new[3], 0.0); p_new[0] = max(p_new[0], 0.0)
        if np.all(np.abs(p_new - p) <= 1e-4 * (np.abs(p) + 1e-9)):
            p = p_new; break
        p = p_new
    A, mu, sg, B = p
    n_peak = A * sg * np.sqrt(2 * np.pi) / bin_width
    n_bkg_3s = B * 6 * sg / bin_width
    return {"mu": float(mu), "sigma": float(sg), "n_peak": float(n_peak), "bkg_per_bin": float(B),
            "bkg_frac": float(n_bkg_3s / max(n_bkg_3s + 0.997 * n_peak, 1e-9)), "n": int(len(x))}


def hit_times(cache: Dict, calib: TQCalibration, key: str = "t_cfd") -> np.ndarray:
    """Hit times in ns relative to the trigger for all cached hits."""
    h = cache["hits"]
    bin_ns = cache["bin_ns"] if "bin_ns" in cache else calib.bin_ns
    return (h[key] * bin_ns[h["cable"], h["atwd"]] - h["launch"] * LAUNCH_OFFSET_NS
            - calib.t0[h["cable"], h["atwd"]])


def hit_charges(cache: Dict, calib: TQCalibration) -> np.ndarray:
    """Calibrated charges (p.e.) of the given cached hits."""
    h = cache["hits"]
    return h["q_adc"] / calib.q1pe[h["cable"], h["atwd"]]


def _group_bounds(key: np.ndarray):
    order = np.argsort(key, kind="stable")
    ks = key[order]
    b = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1], [True])))
    return order, ks, b


def event_time(tau: np.ndarray, ev: np.ndarray, n_events: int, window: float = 10.0) -> np.ndarray:
    """Per-event time: mean of the ToF-corrected hit times within ``window`` ns of
    their 2-ns-binned mode.  Returns an array of length ``n_events`` (NaN if < 4 hits)."""
    T = np.full(n_events, np.nan)
    order, ks, b = _group_bounds(ev)
    tau_s = tau[order]
    for a, c in zip(b[:-1], b[1:]):
        x = tau_s[a:c]
        x = x[np.isfinite(x)]
        if len(x) < 4:
            continue
        lo, hi = np.percentile(x, [2, 98])
        edges = np.arange(lo - 4, hi + 6, 2.0)
        hcount, _ = np.histogram(x, bins=edges)
        hs = np.convolve(hcount, [1, 1, 1], mode="same")
        k = int(hs.argmax())
        c0 = 0.5 * (edges[k] + edges[k + 1])
        sel = np.abs(x - c0) <= window
        if sel.sum() >= 4:
            c0 = x[sel].mean()
            sel = np.abs(x - c0) <= window
            T[ks[a]] = x[sel].mean()
    return T


def residuals_fixed_vertex(cache: Dict, calib: TQCalibration, pmts: PMTTable, vertex: Sequence[float],
                           v_ls: float, v_bo: Optional[float] = None, min_q: float = 0.3):
    """Time residuals ``t - T_event - tof`` for ID primary hits with the vertex fixed.

    Returns a dict with arrays ``res, cable, atwd, q, d, d_ls, d_bo, ev`` (hits with a
    valid event time only)."""
    h = cache["hits"]
    sel = h["primary"] & (h["cable"] < N_ID) & np.isfinite(h["t_cfd"])
    q = hit_charges(cache, calib)
    sel &= q >= min_q
    hs = h[sel]; q = q[sel]
    t = hit_times(cache, calib)[sel]
    r = np.asarray(vertex, dtype=float)
    d_ls, d_bo, u, d = path_lengths(r, pmts.xyz[hs["cable"]])
    tof = d_ls / v_ls + d_bo / (v_bo if v_bo else v_ls)
    tau = t - tof
    T = event_time(tau, hs["ev"], len(cache["events"]))
    Tev = T[hs["ev"]]
    ok = np.isfinite(Tev)
    return {"res": (tau - Tev)[ok], "cable": hs["cable"][ok], "atwd": hs["atwd"][ok], "q": q[ok],
            "d": d[ok], "d_ls": d_ls[ok], "d_bo": d_bo[ok], "ev": hs["ev"][ok], "T": T}


def _mode(x: np.ndarray, width: float = 0.5, smooth: int = 5) -> float:
    lo, hi = np.percentile(x, [1, 99])
    edges = np.arange(lo - width, hi + 2 * width, width)
    hcount, _ = np.histogram(x, bins=edges)
    hs = np.convolve(hcount, np.ones(smooth) / smooth, mode="same")
    k = int(hs.argmax())
    if 0 < k < len(hs) - 1:
        y0, y1, y2 = hs[k - 1], hs[k], hs[k + 1]
        den = y0 - 2 * y1 + y2
        dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        dk = 0.0
    return float(edges[k] + width * (0.5 + dk))


def calibrate_center(cache: Dict, calib: TQCalibration, pmts: PMTTable, v_ls: float = 17.6,
                     min_hits: int = 30, vertex=(0.0, 0.0, 0.0), statistic: str = "mode",
                     n_iter: int = 2, source_like: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
    """T0 and Q0 per (cable, ATWD) from the centre run.

    ``statistic`` is ``"mode"`` (peak of the residual distribution, robust
    against the scintillator tail) or ``"median"``.  Two iterations are done so
    that the event times are recomputed with the new offsets.  Updates
    ``calib`` in place and returns diagnostics."""
    if source_like is not None:
        lo, hi = source_like
        good = (cache["events"]["nhit"] >= lo) & (cache["events"]["nhit"] <= hi)
        cache = dict(cache)
        cache["hits"] = cache["hits"][good[cache["hits"]["ev"]]]
    # --- Q0: mode of the high-gain ADC spectrum per channel -----------------------
    h = cache["hits"]
    sel = h["primary"] & (h["cable"] < N_ID) & (h["q_adc"] > 30) & (h["q_adc"] < 800)
    key = h["cable"][sel].astype(np.int64) * 2 + h["atwd"][sel]
    adc = h["q_adc"][sel]
    order, ks, b = _group_bounds(key)
    adc = adc[order]
    q0 = np.zeros((calib.q1pe.shape[0], 2))
    n_q0 = 0
    for a, c in zip(b[:-1], b[1:]):
        if c - a >= 100:
            k = ks[a]
            q0[k // 2, k % 2] = _mode(adc[a:c], width=10.0, smooth=5)
            n_q0 += 1
    good_q0 = (q0 > 80) & (q0 < 500)
    calib.q1pe[good_q0] = q0[good_q0]
    # --- T0 ---------------------------------------------------------------------------
    t0_total = np.zeros_like(calib.t0)
    for it in range(n_iter):
        rr = residuals_fixed_vertex(cache, calib, pmts, vertex, v_ls)
        key = rr["cable"].astype(np.int64) * 2 + rr["atwd"]
        order, ks, b = _group_bounds(key)
        res = rr["res"][order]
        t0 = np.zeros_like(calib.t0)
        nch = 0
        for a, c in zip(b[:-1], b[1:]):
            if c - a >= min_hits:
                k = ks[a]
                x = res[a:c]
                t0[k // 2, k % 2] = _mode(x) if statistic == "mode" else np.median(x)
                nch += 1
        nz = t0 != 0
        t0[nz] -= t0[nz].mean()
        calib.t0 += t0
        t0_total += t0
    calib.meta["t0_source"] = f"center run {int(cache['run'])} fixed vertex"
    calib.meta["q0_source"] = f"center run {int(cache['run'])}"
    return {"t0": t0_total, "n_t0": nch, "q0": q0, "n_q0": n_q0}


def calibrate_light_yield(cache: Dict, calib: TQCalibration, pmts: PMTTable, v_ls: float, v_bo: Optional[float] = None,
                          vertex=(0.0, 0.0, 0.0), energy_mev: float = 2.506, window=(-15.0, 85.0),
                          dark_window=(-200.0, -40.0), source_like: Optional[Tuple[int, int]] = None,
                          min_events: int = 200) -> Dict[str, np.ndarray]:
    """Per-tube light yield eta_i (p.e./MeV for an event at the vertex, i.e. at the
    centre) and dark hits per event window from the hit probabilities of the
    source events (Detwiler Eq. 4.7-4.9 restricted to one source position)::

        p_i = n_hit_i / N,   mu_i = -ln(1 - p_i),   eta_i = (mu_i - dark_i) / E

    Tubes that are never hit are flagged dead.  Stores ``calib.eta``,
    ``calib.dark`` and ``calib.bad`` and returns diagnostics."""
    if source_like is not None:
        lo, hi = source_like
        good = (cache["events"]["nhit"] >= lo) & (cache["events"]["nhit"] <= hi)
        cache = dict(cache); cache["hits"] = cache["hits"][good[cache["hits"]["ev"]]]
        n_events = int(good.sum())
    else:
        n_events = len(cache["events"])
    rr = residuals_fixed_vertex(cache, calib, pmts, vertex, v_ls, v_bo, min_q=0.0)
    n_events = int(np.isfinite(rr["T"]).sum()) if source_like is None else min(n_events, int(np.isfinite(rr["T"]).sum()))
    if n_events < min_events:
        raise ValueError(f"only {n_events} events for the light-yield calibration")
    inwin = (rr["res"] >= window[0]) & (rr["res"] <= window[1])
    indark = (rr["res"] >= dark_window[0]) & (rr["res"] <= dark_window[1])
    n_hit = np.bincount(rr["cable"][inwin], minlength=N_ID)[:N_ID]
    n_dark = np.bincount(rr["cable"][indark], minlength=N_ID)[:N_ID]
    scale = (window[1] - window[0]) / (dark_window[1] - dark_window[0])
    dark = n_dark / n_events * scale
    p = np.clip(n_hit / n_events, 0.0, 0.999)
    mu = -np.log(1.0 - p)
    eta = np.clip(mu - dark, 0.0, None) / energy_mev
    live = n_hit > 0
    eta[~live] = 0.0
    # charge-based yield for comparison (mean p.e. per event per tube)
    q_sum = np.bincount(rr["cable"][inwin], weights=rr["q"][inwin], minlength=N_ID)[:N_ID]
    eta_q = (q_sum / n_events - dark * np.where(live, np.divide(q_sum, np.maximum(n_hit, 1)), 0.0)) / energy_mev
    calib.eta = eta
    calib.eta_q = np.clip(eta_q, 0.0, None) * live
    calib.dark = dark
    calib.bad[:N_ID] = ~live
    calib.meta["eta_source"] = f"centre run {int(cache['run'])}, {n_events} events"
    return {"eta": eta, "eta_q": eta_q, "dark": dark, "live": live, "n_events": n_events, "mu": mu}


def fit_velocities(caches: Sequence[Dict], zs: Sequence[float], calib: TQCalibration, pmts: PMTTable,
                   v_ls: float = 17.6, v_bo: float = 17.6, n_iter: int = 8, min_q: float = 0.5,
                   only17: bool = True, cell: float = 50.0, min_cell: int = 300, two_media: bool = True,
                   damping: float = 0.6, verbose=False):
    """Effective light speeds from the z-scan.

    For every run the vertex is fixed at the known source position; hits are
    binned in cells of (path in scintillator, path in buffer oil) and the peak
    (mode) of the residual distribution in each cell is regressed as
    ``mode = a + b_ls d_ls + b_bo d_bo`` (or ``a + b d`` if ``two_media`` is
    False); ``1/v`` is updated by ``damping * b`` and the procedure iterated.
    Returns ``(v_ls, v_bo, history)``; history rows are
    ``(v_ls, v_bo, b_ls, b_bo, ncells, rms_ns)``."""
    hist = []
    for it in range(n_iter):
        cells: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for cache, z in zip(caches, zs):
            rr = residuals_fixed_vertex(cache, calib, pmts, (0.0, 0.0, z), v_ls, v_bo, min_q=min_q)
            m = np.ones(len(rr["res"]), dtype=bool)
            if only17:
                m &= rr["cable"] < N_ID17
            i_ls = (rr["d_ls"][m] // cell).astype(int); i_bo = (rr["d_bo"][m] // cell).astype(int)
            key = i_ls * 1000 + i_bo
            order, ks, b = _group_bounds(key)
            res = rr["res"][m][order]
            for a, c in zip(b[:-1], b[1:]):
                cells.setdefault((ks[a] // 1000, ks[a] % 1000), []).append(res[a:c])
        X, y, w = [], [], []
        for (i, j), parts in cells.items():
            x = np.concatenate(parts)
            if len(x) < min_cell:
                continue
            dls, dbo = (i + 0.5) * cell, (j + 0.5) * cell
            X.append([1.0, dls, dbo] if two_media else [1.0, dls + dbo]); y.append(_mode(x)); w.append(np.sqrt(len(x)))
        X = np.array(X); y = np.array(y); w = np.array(w)
        coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        b_ls = coef[1]; b_bo = coef[2] if two_media else coef[1]
        inv_ls = 1.0 / v_ls + damping * b_ls
        inv_bo = 1.0 / v_bo + damping * b_bo
        v_ls_new = 1.0 / inv_ls if inv_ls > 0 else v_ls
        v_bo_new = 1.0 / inv_bo if inv_bo > 0 else v_bo
        resid = y - X @ coef
        hist.append((v_ls, v_bo, b_ls, b_bo, len(y), float(np.sqrt(np.average(resid ** 2, weights=w)))))
        if verbose:
            print(f"  iter {it}: v_ls {v_ls:.3f} -> {v_ls_new:.3f}, v_bo {v_bo:.3f} -> {v_bo_new:.3f}, "
                  f"{len(y)} cells, rms of cell modes {hist[-1][-1]:.3f} ns")
        v_ls, v_bo = v_ls_new, v_bo_new
    return v_ls, v_bo, hist


@dataclass
class TimePDF:
    """Tabulated log-density of the hit-time residual and its derivatives.

    Tables have shape ``(2 tube types, nq, nd, nbins)``; bins are uniform in
    ``[lo, hi)`` with width ``dt``.  Outside the range the edge value is used
    (flat tails, zero derivative)."""

    lo: float
    dt: float
    logp: np.ndarray
    dlogp: np.ndarray
    d2logp: np.ndarray
    q_edges: np.ndarray = field(default_factory=lambda: Q_EDGES.copy())
    d_edges: np.ndarray = field(default_factory=lambda: D_EDGES.copy())
    counts: Optional[np.ndarray] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def nbins(self) -> int:
        """Number of time bins of the PDF tables."""
        return self.logp.shape[-1]

    def bins(self, cable, q, d):
        """Time-bin edges of the PDF tables (ns)."""
        ti = (np.asarray(cable) >= N_ID17).astype(int)
        qi = np.clip(np.searchsorted(self.q_edges, q, side="right") - 1, 0, len(self.q_edges) - 2)
        di = np.clip(np.searchsorted(self.d_edges, d, side="right") - 1, 0, len(self.d_edges) - 2)
        return ti, qi, di

    def evaluate(self, tau, ti, qi, di):
        """Linear interpolation of log p, d log p / d tau, d2 log p / d tau2."""
        x = (np.asarray(tau) - self.lo) / self.dt - 0.5
        x = np.clip(x, 0.0, self.nbins - 1.000001)
        i0 = np.floor(x).astype(int)
        w = x - i0
        i1 = i0 + 1
        lp = self.logp[ti, qi, di, i0] * (1 - w) + self.logp[ti, qi, di, i1] * w
        g = self.dlogp[ti, qi, di, i0] * (1 - w) + self.dlogp[ti, qi, di, i1] * w
        hh = self.d2logp[ti, qi, di, i0] * (1 - w) + self.d2logp[ti, qi, di, i1] * w
        return lp, g, hh

    def save(self, path: str) -> None:
        """Write the PDF tables to an ``.npz`` file."""
        np.savez_compressed(path, lo=self.lo, dt=self.dt, logp=self.logp, dlogp=self.dlogp, d2logp=self.d2logp,
                            q_edges=self.q_edges, d_edges=self.d_edges,
                            counts=self.counts if self.counts is not None else np.zeros(0), meta=str(self.meta))

    @classmethod
    def load(cls, path: str) -> "TimePDF":
        """Read PDF tables written by :meth:`save`."""
        d = np.load(path, allow_pickle=True)
        return cls(float(d["lo"]), float(d["dt"]), d["logp"], d["dlogp"], d["d2logp"], d["q_edges"], d["d_edges"],
                   d["counts"] if d["counts"].size else None, {"note": str(d["meta"])})


def _gauss_smooth(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    n = int(4 * sigma_bins) + 1
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def build_time_pdf(caches: Sequence[Dict], zs: Sequence[float], calib: TQCalibration, pmts: PMTTable,
                   v_ls: float, v_bo: float, lo: float = -40.0, hi: float = 220.0, dt: float = 0.5,
                   smooth_ns: float = 1.0, floor: float = 1e-5, q_edges=Q_EDGES, d_edges=D_EDGES,
                   min_q: float = 0.3, min_count: int = 2000, source_like: Optional[Tuple[int, int]] = None) -> TimePDF:
    """Empirical residual densities from the z-scan with the vertices fixed at the
    known source positions.  Cells with fewer than ``min_count`` hits fall back to
    the density integrated over distance (then over charge)."""
    nq, nd = len(q_edges) - 1, len(d_edges) - 1
    edges = np.arange(lo, hi + dt / 2, dt)
    nb = len(edges) - 1
    counts = np.zeros((2, nq, nd, nb))
    for cache, z in zip(caches, zs):
        if source_like is not None:
            lo_n, hi_n = source_like
            good = (cache["events"]["nhit"] >= lo_n) & (cache["events"]["nhit"] <= hi_n)
            cache = dict(cache); cache["hits"] = cache["hits"][good[cache["hits"]["ev"]]]
        rr = residuals_fixed_vertex(cache, calib, pmts, (0.0, 0.0, z), v_ls, v_bo, min_q=min_q)
        ti = (rr["cable"] >= N_ID17).astype(int)
        qi = np.clip(np.searchsorted(q_edges, rr["q"], side="right") - 1, 0, nq - 1)
        di = np.clip(np.searchsorted(d_edges, rr["d"], side="right") - 1, 0, nd - 1)
        bi = np.floor((rr["res"] - lo) / dt).astype(int)
        m = (bi >= 0) & (bi < nb)
        np.add.at(counts, (ti[m], qi[m], di[m], bi[m]), 1)
    # fallbacks for sparse cells
    total_q = counts.sum(axis=2, keepdims=True)          # integrated over distance
    total_t = counts.sum(axis=(1, 2), keepdims=True)     # integrated over q and d
    filled = np.where(counts.sum(-1, keepdims=True) >= min_count, counts,
                      np.where(np.broadcast_to(total_q.sum(-1, keepdims=True), counts.shape) >= min_count,
                               np.broadcast_to(total_q, counts.shape), np.broadcast_to(total_t, counts.shape)))
    sig = smooth_ns / dt
    logp = np.empty_like(filled); dlogp = np.empty_like(filled); d2logp = np.empty_like(filled)
    for idx in np.ndindex(filled.shape[:-1]):
        y = _gauss_smooth(filled[idx], sig)
        if y.max() <= 0:                      # no hits at all for this tube type (e.g. 20-inch tubes off in 2002)
            y = np.full_like(y, 1.0)          # flat density: log p constant, zero derivatives
        y = y / max(y.sum() * dt, 1e-300)
        y = np.maximum(y, floor * y.max())
        lp = np.log(y)
        g = np.gradient(lp, dt)
        g = _gauss_smooth(g, sig)
        h2 = _gauss_smooth(np.gradient(g, dt), sig)
        logp[idx] = lp; dlogp[idx] = g; d2logp[idx] = h2
    pdf = TimePDF(lo, dt, logp, dlogp, d2logp, np.asarray(q_edges, float), np.asarray(d_edges, float), counts,
                  {"v_ls": v_ls, "v_bo": v_bo, "runs": [int(c["run"]) for c in caches], "smooth_ns": smooth_ns})
    return pdf


# ---------------------------------------------------------------------------
# geometry survey: per-tube position / time offsets from the z-scan
# ---------------------------------------------------------------------------
def survey_positions(caches: Sequence[Dict], zs: Sequence[float], calib: TQCalibration, pmts: PMTTable,
                     v_ls: float, v_bo: Optional[float] = None, min_hits: int = 30, sigma_prior_cm: float = 30.0,
                     min_q: float = 0.5) -> Dict[str, np.ndarray]:
    """Fit a position offset dP_i (cm) and a time offset dT_i (ns) for every inner tube
    from the per-run modes m_ik of its time residual (vertex fixed at the source)::

        m_ik = dT_i + u_ik . dP_i / v_eff

    where u_ik is the unit vector from source k to tube i.  A Gaussian prior of
    ``sigma_prior_cm`` on dP regularises tubes with little lever arm (near the
    axis the direction to the source hardly changes with z).  This is a
    *diagnostic* of the mechanical tolerances (sphere shape, bolt pattern); it
    uses the whole scan.  Returns dP (N_ID x 3), dT, their uncertainties, the
    number of runs per tube and the residual rms of the fit."""
    v_bo = v_bo if v_bo else v_ls
    P = pmts.xyz[:N_ID]
    modes = np.full((len(caches), N_ID), np.nan)
    counts = np.zeros((len(caches), N_ID), dtype=int)
    for k, (cache, z) in enumerate(zip(caches, zs)):
        rr = residuals_fixed_vertex(cache, calib, pmts, (0.0, 0.0, z), v_ls, v_bo, min_q=min_q)
        order, ks, b = _group_bounds(rr["cable"].astype(np.int64))
        res = rr["res"][order]
        for a, c in zip(b[:-1], b[1:]):
            if c - a >= min_hits:
                modes[k, ks[a]] = _mode(res[a:c])
                counts[k, ks[a]] = c - a
    dP = np.zeros((N_ID, 3)); dT = np.zeros(N_ID); err = np.full((N_ID, 4), np.nan)
    nrun = (counts > 0).sum(axis=0); rms = np.full(N_ID, np.nan)
    src = np.array([[0.0, 0.0, z] for z in zs])
    prior = np.diag([0.0, 1.0 / sigma_prior_cm ** 2, 1.0 / sigma_prior_cm ** 2, 1.0 / sigma_prior_cm ** 2])
    for i in range(N_ID):
        ok = np.isfinite(modes[:, i])
        if ok.sum() < 4:
            continue
        d = P[i][None, :] - src[ok]
        dist = np.linalg.norm(d, axis=1)
        u = d / dist[:, None]
        # effective speed along the path (two media)
        d_ls, d_bo, _, _ = path_lengths_multi(src[ok], P[i])
        veff = dist / (d_ls / v_ls + d_bo / v_bo)
        Jm = np.column_stack([np.ones(ok.sum()), u / veff[:, None]])
        w = np.sqrt(counts[ok, i])
        A = (Jm * w[:, None]).T @ (Jm * w[:, None]) / np.mean(w ** 2) + prior
        bvec = (Jm * w[:, None]).T @ (modes[ok, i] * w) / np.mean(w ** 2)
        x = np.linalg.solve(A, bvec)
        dT[i] = x[0]; dP[i] = x[1:]
        resid = modes[ok, i] - Jm @ x
        rms[i] = np.sqrt(np.mean(resid ** 2))
        # uncertainties: scale the inverse normal matrix by the residual variance
        s2 = max(np.sum(w ** 2 * resid ** 2) / max(ok.sum() - 4, 1) / np.mean(w ** 2), 1e-6)
        err[i] = np.sqrt(np.diag(np.linalg.inv(A)) * s2)
    return {"dP": dP, "dT": dT, "err": err, "nrun": nrun, "rms": rms, "modes": modes, "counts": counts}


def path_lengths_multi(sources: np.ndarray, p: np.ndarray, radius: float = BALLOON_RADIUS_CM):
    """Scintillator / buffer-oil path lengths from several source positions to one tube."""
    out_ls = np.empty(len(sources)); out_bo = np.empty(len(sources))
    for k, s in enumerate(sources):
        d_ls, d_bo, u, d = path_lengths(s, p[None, :], radius)
        out_ls[k] = d_ls[0]; out_bo[k] = d_bo[0]
    return out_ls, out_bo, None, None


def deformation_vs_theta(pmts: PMTTable, dP: np.ndarray, nbins: int = 12, good: Optional[np.ndarray] = None):
    """Mean radial and vertical offset of the tubes in bins of cos(theta) (theta from the +z axis)."""
    P = pmts.xyz[:N_ID]
    r = np.linalg.norm(P, axis=1)
    rhat = P / r[:, None]
    dr = np.einsum("ij,ij->i", dP, rhat)
    cost = P[:, 2] / r
    if good is None:
        good = np.ones(N_ID, dtype=bool)
    edges = np.linspace(-1, 1, nbins + 1)
    idx = np.clip(np.digitize(cost, edges) - 1, 0, nbins - 1)
    rows = []
    for b in range(nbins):
        m = (idx == b) & good
        if m.sum() >= 5:
            rows.append((0.5 * (edges[b] + edges[b + 1]), m.sum(), dr[m].mean(), dr[m].std() / np.sqrt(m.sum()),
                         dP[m, 2].mean(), np.hypot(dP[m, 0], dP[m, 1]).mean()))
    return np.array(rows), dr
