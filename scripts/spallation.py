#!/usr/bin/env python3
"""Muon-induced spallation neutrons: muon tracks, then the events that follow each muon.

For every physics event the TQ is computed and the standard muon selection
(``muon.is_muon``: Q17 >= 10 000 p.e., or Q17 >= 500 p.e. with N200_OD >= 5) is
applied.  Muons get a track fit (`MuonTrackFitter`).  Every non-muon event within
``--window`` after the most recent muon is fully reconstructed (vertex and energy
with the configured calibration and time densities) and stored with the time since
the muon and the distance of its vertex to the track; a neutron captured on a proton
gives a 2.22 MeV gamma (E_vis 2.19 MeV) with a mean capture time of ~207 us.

    python scripts/spallation.py /hsm/.../run_001545_000000_00000[1-6].sf -o ~/cache/spallation/spall_1545.npz
    python scripts/spallation.py --analyze ~/cache/spallation/spall_*.npz -o ~/cache/spallation/spallation

Output ``.npz``: ``muons`` (MUON_DTYPE + ``dt_prev_us``, ``retrigger``, ``n_after``) and
``cands`` (EVENT_DTYPE + ``dt_us``, ``muon``, ``dist_track``, ``mu_q17``, ``mu_l_ls``,
``mu_delta_q``, ``mu_converged``).  ``--analyze`` fits the capture time and makes the plots.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from claudland import config as _config

CFG = _config.get()

import numpy as np

from claudland.reco import EventReconstructor, EVENT_DTYPE
from claudland.banks import decode_header, decode_hit_header
from claudland.geometry import PMTTable, N_ID, N_ID17
from claudland.calib import TQCalibration
from claudland.vertex import VertexFitter
from claudland.energy import EnergyEstimator
from claudland.muon import (MuonTrackFitter, MuonChargeModel, muon_hits, is_muon, n200_od, muon_row, MUON_DTYPE)

TICK_US = 0.025                       # 40 MHz clock
MUON_EXTRA = [("dt_prev_us", np.float64), ("retrigger", np.bool_), ("n_after", np.int16)]
CAND_EXTRA = [("dt_us", np.float64), ("muon", np.int32), ("dist_track", np.float32), ("mu_q17", np.float32),
              ("mu_l_ls", np.float32), ("mu_delta_q", np.float32), ("mu_converged", np.bool_)]
MUON_OUT_DTYPE = np.dtype(MUON_DTYPE.descr + MUON_EXTRA)
CAND_DTYPE = np.dtype(EVENT_DTYPE.descr + CAND_EXTRA)
N_CAPTURE_EVIS_MEV = 2.189            # 2.22457 MeV gamma with the E_vis/E_real tables


def build_reconstructor(files, calib_path, pdf_path, joint="hit", gains=(0, 1, 2), verbose=True):
    """EventReconstructor with the configured calibration and the joint ML vertex fitter (as reco_run.py)."""
    pmts = PMTTable.load()
    calib = TQCalibration.load(calib_path) if calib_path else None
    v_ls = float(calib.meta["v_ls"]) if calib and "v_ls" in calib.meta else 17.6
    v_bo = float(calib.meta["v_bo"]) if calib and "v_bo" in calib.meta else None
    fitter = VertexFitter(pmts, v_ls=v_ls, v_bo=v_bo)
    energy = EnergyEstimator(pmts, eta=getattr(calib, "eta", None), eta_charge=getattr(calib, "eta_q", None),
                             dark_per_tube=getattr(calib, "dark", None))
    if calib is not None and calib.eta is not None:
        energy.live = calib.eta[:N_ID] > 0
    if pdf_path:
        from claudland.vertex_ml import MLVertexFitter, ChargeTimeVertexFitter
        from claudland.zscan import TimePDF
        pdf = TimePDF.load(pdf_path)
        if joint:
            fitter = ChargeTimeVertexFitter(pmts, pdf, v_ls, v_bo, energy_model=energy, charge_model=joint, prefit=fitter)
        else:
            fitter = MLVertexFitter(pmts, pdf, v_ls, v_bo, prefit=fitter)
    rec = EventReconstructor(files, pmts=pmts, calib=calib, vertex=fitter, energy=energy, gains=gains, verbose=verbose)
    return rec, pmts


def scale_energies(tab, calib):
    """Apply the energy-scale factors stored in the calibration (as reco_run.py does)."""
    if calib is None:
        return
    for key in ("e_charge", "e_hit"):
        sc = calib.meta.get(f"scale_{key}")
        if sc:
            tab[key] *= float(sc)


def process(args):
    """Scan the files: tag and fit muons, reconstruct the events in the window after each muon."""
    rec, pmts = build_reconstructor(args.files, args.calib, args.ml_pdf, joint=args.joint, verbose=not args.quiet)
    rec.prepare()
    live17 = rec.energy_estimator.live[:N_ID17] if rec.energy_estimator.live is not None else None
    cm = MuonChargeModel(pmts, index=1.5, live=live17)
    mufit = MuonTrackFitter(pmts, index=1.5, charge_model=cm, n_scan=args.n_scan)
    window_ticks = int(args.window / TICK_US)
    retrig_ticks = int(args.retrigger / TICK_US)

    muons, cands = [], []
    last_ts = None; last_track = None; last_row = None; last_idx = -1
    n_events = n_muon = n_retrig = 0
    t_start = time.time()
    for ev in rec.physics_events(max_events=args.max_events):
        n_events += 1
        h = decode_header(ev["Header"])
        ts = int(h.timestamp)
        r = rec.reconstruct(ev, fit_vertex=False)
        mh = muon_hits(r.hits)
        idm = mh["cable"] < N_ID
        q17 = float(mh["q"][mh["cable"] < N_ID17].sum())
        n_od = n200_od(mh["t"][~idm])
        if is_muon(q17, n_od):
            dt_prev = (ts - last_ts) * TICK_US if last_ts is not None else np.nan
            retrig = last_ts is not None and (ts - last_ts) < retrig_ticks and q17 < 10000.0
            od = mh[~idm]
            q5 = float(od["q"][(od["cable"] >= 2120) & (od["cable"] <= 2125)].sum())
            track = None
            # outer-detector muons (little ID charge, few ID hits) cannot have crossed the inner detector;
            # their track fit never converges and costs ~2 s each, so skip it unless asked for
            od_only = q17 < 10000.0 and int(idm.sum()) < args.id_muon_nhit
            if (not retrig or args.fit_retriggers) and (not od_only or args.fit_od_muons):
                track = mufit.fit(mh["cable"][idm], mh["t"][idm], mh["q"][idm], q17=q17,
                                  cable_od=od["cable"], t_od=od["t"], q_5inch=q5)
            row = np.zeros(1, dtype=MUON_OUT_DTYPE)[0]
            base = muon_row(h, ev.index, mh, track, pmts)
            for name in MUON_DTYPE.names:
                row[name] = base[name]
            row["dt_prev_us"] = dt_prev; row["retrigger"] = retrig
            muons.append(row)
            if retrig:
                n_retrig += 1
            else:
                n_muon += 1
                last_ts = ts; last_track = track; last_row = row; last_idx = len(muons) - 1
                if not args.quiet:
                    print(f"muon ev {ev.index:6d} nhit={row['nhit']:4d} Q17={q17:9.0f} N200OD={n_od:3d} "
                          f"{track.summary() if track is not None else ''}", flush=True)
            continue
        if last_ts is None or (ts - last_ts) > window_ticks:
            continue
        hh = decode_hit_header(ev["HitHeader"])
        if hh.nhit < args.cand_min_nhit:          # afterpulse re-triggers etc.: far below a 2.2 MeV gamma
            muons[last_idx]["n_after"] += 1
            continue
        # a non-muon event shortly after a muon: full reconstruction
        r2 = rec.reconstruct(ev)
        row = np.zeros(1, dtype=CAND_DTYPE)[0]
        base = r2.record()
        for name in EVENT_DTYPE.names:
            row[name] = base[name]
        row["dt_us"] = (ts - last_ts) * TICK_US
        row["muon"] = last_idx
        row["mu_q17"] = last_row["q17"]; row["mu_l_ls"] = last_row["l_ls"]; row["mu_delta_q"] = last_row["delta_q"]
        row["mu_converged"] = bool(last_row["converged"])
        if last_track is not None and np.isfinite(last_track.t0) and row["vertex_ok"]:
            row["dist_track"] = float(last_track.distance(np.array([row["x"], row["y"], row["z"]], dtype=float))[0])
        else:
            row["dist_track"] = np.nan
        cands.append(row)
        muons[last_idx]["n_after"] += 1
        if n_events % 2000 == 0 and not args.quiet:
            print(f"  {n_events} events, {n_muon} muons, {len(cands)} candidates, {time.time() - t_start:.0f} s", flush=True)
    muons = np.array(muons, dtype=MUON_OUT_DTYPE) if muons else np.zeros(0, MUON_OUT_DTYPE)
    cands = np.array(cands, dtype=CAND_DTYPE) if cands else np.zeros(0, CAND_DTYPE)
    scale_energies(cands, rec.calib)
    live_s = 0.0
    if n_events:
        live_s = float(rec.timing.get("run", 0.0))
    print(f"{n_events} physics events, {n_muon} muons ({n_retrig} re-triggers within {args.retrigger:.0f} us), "
          f"{len(cands)} events within {args.window:.0f} us after a muon  [{time.time() - t_start:.0f} s]")
    if len(muons):
        m = muons[~muons["retrigger"]]
        print(f"  muon tracks converged {int(m['converged'].sum())}/{len(m)}, through the LS {int((m['l_ls'] > 0).sum())}")
    if args.output:
        run_no = rec.run_header.run if rec.run_header else (int(muons["run"][0]) if len(muons) else -1)   # continuation files have no RunHeader
        np.savez_compressed(args.output, muons=muons, cands=cands, run=run_no,
                            window_us=args.window, n_events=n_events,
                            t_first=float(muons["unix_time"].min()) if len(muons) else np.nan,
                            t_last=float(muons["unix_time"].max()) if len(muons) else np.nan)
        print(f"written {args.output}")


# ---------------------------------------------------------------------------
# analysis: capture-time fit and plots
# ---------------------------------------------------------------------------
def fit_exponential_flat(counts, edges, tau_grid=None):
    """Poisson maximum-likelihood fit of ``A exp(-t/tau) + B`` per bin to a histogram.

    Returns ``tau, tau_err, A, B, n_signal, n_background, nll`` (tau in the units
    of *edges*; A and B per bin at the bin centres).  The profile likelihood
    in tau is scanned on ``tau_grid``; A and B are optimised by Newton steps."""
    c = 0.5 * (edges[:-1] + edges[1:]); w = np.diff(edges)
    y = counts.astype(float)
    if tau_grid is None:
        tau_grid = np.arange(50.0, 600.0, 1.0)
    best = None; prof = []
    for tau in tau_grid:
        e = np.exp(-c / tau)
        # start: B from the last quarter of the bins, A from the first bins
        B = max(np.mean(y[-max(len(y) // 4, 1):]), 1e-3)
        A = max((y[0] - B) / e[0], 1e-3)
        for _ in range(60):
            mu = A * e + B
            mu = np.maximum(mu, 1e-9)
            gA = np.sum(e * (1 - y / mu)); gB = np.sum(1 - y / mu)
            hAA = np.sum(e * e * y / mu ** 2); hAB = np.sum(e * y / mu ** 2); hBB = np.sum(y / mu ** 2)
            det = hAA * hBB - hAB * hAB
            if det <= 0:
                break
            dA = (hBB * gA - hAB * gB) / det; dB = (hAA * gB - hAB * gA) / det
            A = max(A - dA, 1e-6); B = max(B - dB, 1e-6)
            if abs(dA) < 1e-6 * (A + 1) and abs(dB) < 1e-6 * (B + 1):
                break
        mu = np.maximum(A * e + B, 1e-9)
        nll = float(np.sum(mu - y * np.log(mu)))
        prof.append(nll)
        if best is None or nll < best[0]:
            best = (nll, tau, A, B)
    prof = np.array(prof)
    nll0, tau, A, B = best
    inside = tau_grid[prof <= nll0 + 0.5]
    tau_err = 0.5 * (inside.max() - inside.min()) if len(inside) > 1 else np.nan
    e = np.exp(-c / tau)
    return {"tau": tau, "tau_err": tau_err, "A": A, "B": B, "n_signal": float(np.sum(A * e)),
            "n_background": float(B * len(c)), "nll": nll0, "centres": c, "model": A * e + B, "tau_grid": tau_grid, "profile": prof}


def analyze(args):
    """Merge the .npz files, fit the capture time, plot."""
    files = sorted(set(sum([glob.glob(f) for f in args.files], [])))
    muons = []; cands = []; n_events = 0; window = None; runs = []; live = 0.0
    for f in files:
        d = np.load(f, allow_pickle=True)
        muons.append(d["muons"]); cands.append(d["cands"]); n_events += int(d["n_events"]); runs.append(int(d["run"]))
        window = float(d["window_us"])
        if np.isfinite(d["t_first"]) and np.isfinite(d["t_last"]):
            live += float(d["t_last"] - d["t_first"])
    muons = np.concatenate(muons); cands = np.concatenate(cands)
    real = muons[~muons["retrigger"]]
    print(f"{len(files)} files, runs {sorted(set(runs))}: {n_events} physics events, {len(real)} muons "
          f"({int(muons['retrigger'].sum())} re-triggers), {len(cands)} events within {window:.0f} us after a muon; "
          f"~{live:.0f} s between first and last muon -> muon rate {len(real) / max(live, 1):.2f} Hz")
    ok = cands["vertex_ok"] & np.isfinite(cands["e_hit"])
    e = cands["e_hit"]
    parent = muons[cands["muon"]]
    if args.parent == "ls":
        ok &= parent["converged"] & (parent["l_ls"] > 0)
    elif args.parent == "id":
        ok &= (parent["q17"] >= 10000.0) | parent["converged"]
    fid = ok & (cands["r"] < args.r_max)
    sel = fid & (e >= args.e_lo) & (e <= args.e_hi) & (cands["dt_us"] >= args.dt_min) & (cands["dt_us"] <= window)
    print(f"neutron-candidate selection: {args.e_lo}-{args.e_hi} MeV, r < {args.r_max:.0f} cm, "
          f"{args.dt_min:.0f} us <= dt <= {window:.0f} us: {int(sel.sum())} events")
    edges = np.arange(args.dt_min, window + 1e-6, args.bin_us)
    counts, _ = np.histogram(cands["dt_us"][sel], bins=edges)
    fit = fit_exponential_flat(counts, edges)
    conv = real["converged"]
    ls = conv & (real["l_ls"] > 0)
    n_parent = {"ls": int(ls.sum()), "id": int(((real["q17"] >= 10000.0) | conv).sum()), "all": len(real)}[args.parent]
    print(f"capture time: tau = {fit['tau']:.1f} +- {fit['tau_err']:.1f} us  (expected ~207 us); "
          f"neutrons {fit['n_signal']:.0f}, flat background {fit['n_background']:.0f} events "
          f"({fit['B'] / (args.bin_us * 1e-6) / max(n_parent, 1):.2f} Hz accidental rate in the selection per parent muon)")
    print(f"parent muons ({args.parent}): {n_parent}; neutrons per parent muon {fit['n_signal'] / max(n_parent, 1):.3f}; "
          f"LS muons {int(ls.sum())}, mean L_LS {real['l_ls'][ls].mean():.0f} cm")
    # electronics recovery after the muon: multiplicity of the follow-up events versus time since the muon
    ls_parent = parent["converged"] & (parent["l_ls"] > 0)
    rec_edges = np.array([0, 5, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000.0])
    print("recovery after LS muons (all follow-up events): dt bin [us] -> events, median nhit, fraction with nhit >= 300")
    rec_frac = []
    for a, b in zip(rec_edges[:-1], rec_edges[1:]):
        m = ls_parent & (cands["dt_us"] >= a) & (cands["dt_us"] < b)
        nh = cands["nhit"][m]
        frac = float((nh >= 300).mean()) if m.any() else np.nan
        rec_frac.append(frac)
        print(f"  {a:6.0f}-{b:6.0f}: {int(m.sum()):6d} {np.median(nh) if m.any() else np.nan:6.0f} {frac:6.2f}")
    # late-time control sample for the energy / distance spectra
    late = fid & (cands["dt_us"] > max(6 * fit["tau"], 0.5 * window)) & (cands["dt_us"] <= window)
    early = fid & (cands["dt_us"] >= args.dt_min) & (cands["dt_us"] <= 3 * fit["tau"])
    w_late = (3 * fit["tau"] - args.dt_min) / max(window - max(6 * fit["tau"], 0.5 * window), 1e-9)
    if args.output:
        np.savez_compressed(args.output + ".npz", muons=muons, cands=cands, dt_edges=edges, dt_counts=counts,
                            tau=fit["tau"], tau_err=fit["tau_err"], n_signal=fit["n_signal"], n_background=fit["n_background"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(19, 9))
        ax = axs[0, 0]
        ax.errorbar(fit["centres"], counts, yerr=np.sqrt(np.maximum(counts, 1)), fmt="o", ms=3, color="k", label="data")
        ax.plot(fit["centres"], fit["model"], color="C3", label=f"A e^(-t/tau) + B, tau = {fit['tau']:.0f} +- {fit['tau_err']:.0f} us")
        ax.axhline(fit["B"], color="C0", ls="--", lw=0.8, label=f"accidentals B = {fit['B']:.1f} / bin")
        ax.set_yscale("log"); ax.set_xlabel("time after muon [us]"); ax.set_ylabel(f"events / {args.bin_us:.0f} us")
        ax.set_title(f"{args.e_lo}-{args.e_hi} MeV, r < {args.r_max:.0f} cm: {fit['n_signal']:.0f} neutrons"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axs[0, 1]
        eb = np.arange(0.0, 6.01, 0.1)
        he, _ = np.histogram(e[early], bins=eb); hl, _ = np.histogram(e[late], bins=eb)
        ax.step(eb[:-1], he, where="post", color="C3", label=f"{args.dt_min:.0f}-{3 * fit['tau']:.0f} us after muon")
        ax.step(eb[:-1], hl * w_late, where="post", color="C0", label="late window (scaled)")
        ax.axvline(N_CAPTURE_EVIS_MEV, color="k", lw=0.7, ls=":", label="2.22 MeV gamma (E_vis 2.19)")
        ax.set_xlabel("E_vis (hit pattern) [MeV]"); ax.set_ylabel("events / 0.1 MeV"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axs[1, 0]
        db = np.arange(0.0, 1000.0, 50.0)
        dsel = sel & np.isfinite(cands["dist_track"]) & cands["mu_converged"]
        dlate = late & (e >= args.e_lo) & (e <= args.e_hi) & np.isfinite(cands["dist_track"]) & cands["mu_converged"]
        hd, _ = np.histogram(cands["dist_track"][dsel & (cands["dt_us"] <= 3 * fit["tau"])], bins=db)
        hdl, _ = np.histogram(cands["dist_track"][dlate], bins=db)
        ax.step(db[:-1], hd, where="post", color="C3", label="candidates, dt < 3 tau")
        ax.step(db[:-1], hdl * w_late, where="post", color="C0", label="late window (scaled)")
        ax.set_xlabel("distance of the vertex to the muon track [cm]"); ax.set_ylabel("events / 50 cm"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axs[0, 2]
        m = ls_parent
        ax.scatter(cands["dt_us"][m], cands["nhit"][m], s=3, alpha=0.4, color="C1")
        ax.set_xscale("log"); ax.set_xlabel("time after LS muon [us]"); ax.set_ylabel("ID hits of the follow-up event")
        ax.axhline(300, color="k", lw=0.6, ls=":"); ax.axvline(args.dt_min, color="C3", lw=0.8, ls="--", label="start of the fit")
        ax.set_title("electronics recovery: multiplicity after the muon"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axs[1, 2]
        rc = 0.5 * (rec_edges[:-1] + rec_edges[1:])
        ax.step(rec_edges[:-1], rec_frac, where="post", color="C1")
        ax.set_xscale("log"); ax.set_xlabel("time after LS muon [us]"); ax.set_ylabel("fraction of follow-up events with nhit >= 300")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax = axs[1, 1]
        nsig_per_mu = np.bincount(cands["muon"][sel], minlength=len(muons))
        q = real["q17"]; okq = q > 0
        ax.scatter(q[okq], nsig_per_mu[~muons["retrigger"]][okq] + 0.05 * np.random.default_rng(0).standard_normal(okq.sum()),
                   s=6, alpha=0.5, color="C2")
        ax.set_xscale("log"); ax.set_xlabel("muon Q17 [p.e.]"); ax.set_ylabel("neutron candidates after the muon")
        ax.grid(alpha=0.3); ax.set_title(f"{len(real)} muons, {int(ls.sum())} through the LS")
        fig.suptitle(f"spallation neutrons, runs {sorted(set(runs))}: capture time {fit['tau']:.1f} +- {fit['tau_err']:.1f} us")
        fig.tight_layout()
        if args.output:
            fig.savefig(args.output + ".png", dpi=110); print("plot:", args.output + ".png")
    except ImportError:
        pass


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="raw files of one run (or .npz outputs with --analyze)")
    ap.add_argument("-o", "--output", help="output .npz (processing) or basename for .npz/.png (--analyze)")
    ap.add_argument("--analyze", action="store_true", help="merge .npz outputs, fit the capture time, plot")
    ap.add_argument("--calib", default=str(CFG.optional("tq") or ""))
    ap.add_argument("--ml-pdf", default=str(CFG.optional("time_pdf") or ""))
    ap.add_argument("--joint", choices=["hit", "poisson", ""], default="hit")
    ap.add_argument("--window", type=float, default=2000.0, help="time window after a muon (us)")
    ap.add_argument("--retrigger", type=float, default=100.0,
                    help="a muon-like event with Q17 < 10000 p.e. this soon (us) after a muon is a re-trigger, not a new muon")
    ap.add_argument("--fit-retriggers", action="store_true")
    ap.add_argument("--fit-od-muons", action="store_true", help="also fit tracks for muons with Q17 < 10000 p.e. and few ID hits")
    ap.add_argument("--id-muon-nhit", type=int, default=500, help="ID hits below which a Q17 < 10000 p.e. muon counts as OD-only")
    ap.add_argument("--cand-min-nhit", type=int, default=300,
                    help="reconstruct follow-up events only above this ID multiplicity (a 2.2 MeV gamma gives ~550 hits)")
    ap.add_argument("--n-scan", type=int, default=120)
    ap.add_argument("-n", "--max-events", type=int, default=None)
    ap.add_argument("--e-lo", type=float, default=1.8, help="E_vis window of the neutron candidates (MeV)")
    ap.add_argument("--e-hi", type=float, default=2.6)
    ap.add_argument("--r-max", type=float, default=600.0, help="fiducial radius of the candidates (cm)")
    ap.add_argument("--dt-min", type=float, default=150.0,
                    help="start of the capture-time fit (us); the front-end channels are busy with the muon waveforms for "
                         "~100 us, during which a 2.2 MeV gamma registers only a few hits")
    ap.add_argument("--parent", choices=["all", "ls", "id"], default="ls",
                    help="parent muons used: all tagged muons, only tracks through the scintillator (l_ls > 0, default), "
                         "or all inner-detector muons (Q17 >= 10000 p.e. or converged track)")
    ap.add_argument("--bin-us", type=float, default=20.0)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(args)
    else:
        process(args)


if __name__ == "__main__":
    main()
