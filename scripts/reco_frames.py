#!/usr/bin/env python3
"""Reconstruct every physics event of a run into a pandas frame (Parquet) for analysis.

Per event: header bookkeeping, multiplicities and charges, the muon selection, the joint
likelihood vertex and energies (non-muons), the fitted muon track (muons), the time since
the previous event and since the previous muon, and the distance of the vertex to that
muon's track.  See :mod:`claudland.frames` for the column list and a loader.

    python scripts/reco_frames.py 1550 --file-range 1-4 -o ~/cache/frames/run001550/reco_001550_f001-004.parquet
    python scripts/reco_frames.py /hsm/.../run_001550_000000_000001.sf -o out.parquet -n 500

    python scripts/farm.py frames 1550 --file-range 1-30 --tag f001-030      # one LSF job per run

The first argument is a run number (files found through claudland.toml) or a list of files.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from claudland import config as _config

CFG = _config.get()

import numpy as np
import pandas as pd

from claudland.reco import EventReconstructor
from claudland.banks import decode_header
from claudland.geometry import PMTTable, N_ID, N_ID17
from claudland.calib import TQCalibration
from claudland.vertex import VertexFitter
from claudland.energy import EnergyEstimator
from claudland.muon import MuonTrackFitter, MuonChargeModel, muon_hits, is_muon, n200_od, DQDX_LS, DQDX_BO

TICK_US = 0.025

MU_COLS = ["mu_ex", "mu_ey", "mu_ez", "mu_xx", "mu_xy", "mu_xz", "mu_ux", "mu_uy", "mu_uz", "mu_t0", "mu_cos_zenith",
           "mu_impact", "mu_l_ls", "mu_l_bo", "mu_sigma_t", "mu_frac_used", "mu_delta_q", "mu_charge_scale", "mu_n_used"]


def build_reconstructor(files, calib_path, pdf_path, joint="hit", verbose=True):
    """EventReconstructor with the configured calibration and the joint ML vertex fitter."""
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
    rec = EventReconstructor(files, pmts=pmts, calib=calib, vertex=fitter, energy=energy, gains=(0, 1, 2), verbose=verbose)
    return rec, pmts, calib


def vertex_energy(rec, hits):
    """Vertex + energy of a hit array (the body of EventReconstructor.reconstruct without the decoding)."""
    hits_id = hits[hits["cable"] < N_ID]
    p = hits_id["primary"]
    if p.sum() < 4:
        return None, None
    vertex = rec.vertex_fitter.fit(hits_id["cable"][p], hits_id["t"][p], hits_id["q"][p])
    energy = None
    if np.isfinite(vertex.x):
        tau = rec.vertex_fitter.residuals(vertex, hits_id["cable"][p], hits_id["t"][p])
        energy = rec.energy_estimator.estimate(hits_id["cable"][p], tau, hits_id["q"][p], vertex.xyz)
    return vertex, energy


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="run number, or the run's files in order")
    ap.add_argument("-o", "--output", required=True, help="output .parquet")
    ap.add_argument("--files", type=int, default=None, help="with a run number: first N files")
    ap.add_argument("--file-range", default=None, metavar="A-B", help="with a run number: files A..B (1-based)")
    ap.add_argument("--calib", default=str(CFG.optional("tq") or ""))
    ap.add_argument("--ml-pdf", default=str(CFG.optional("time_pdf") or ""))
    ap.add_argument("--joint", choices=["hit", "poisson", ""], default="hit")
    ap.add_argument("--retrigger", type=float, default=100.0, help="re-trigger window after a muon (us)")
    ap.add_argument("--n-scan", type=int, default=120, help="grid points of the muon track scan")
    ap.add_argument("--fit-od-muons", action="store_true", help="also fit tracks of outer-detector-only muons")
    ap.add_argument("-n", "--max-events", type=int, default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if len(args.source) == 1 and args.source[0].isdigit():
        run = int(args.source[0])
        files = [str(f) for f in CFG.run_files(run)]
        if args.file_range:
            a, b = (int(v) for v in args.file_range.split("-"))
            files = files[a - 1:b]
        elif args.files:
            files = files[:args.files]
        if not files:
            sys.exit(f"no files for run {run} below {[str(d) for d in CFG.data_dirs]}")
    else:
        files = args.source
    file_numbers = {}
    for f in files:                       # sub-run file number from the name run_RRRRRR_SSSSSS_FFFFFF.sf
        try:
            file_numbers[f] = int(os.path.basename(f).split("_")[3].split(".")[0])
        except (IndexError, ValueError):
            file_numbers[f] = -1

    rec, pmts, calib = build_reconstructor(files, args.calib, args.ml_pdf, joint=args.joint, verbose=not args.quiet)
    rec.prepare()
    live17 = rec.energy_estimator.live[:N_ID17] if rec.energy_estimator.live is not None else None
    mufit = MuonTrackFitter(pmts, index=1.5, charge_model=MuonChargeModel(pmts, index=1.5, live=live17), n_scan=args.n_scan)
    scale = {k: float(calib.meta.get(f"scale_{k}", 1.0) or 1.0) for k in ("e_hit", "e_charge")} if calib else {"e_hit": 1.0, "e_charge": 1.0}
    retrig_ticks = int(args.retrigger / TICK_US)

    rows = []
    prev_ts = None; mu_ts = None; mu_event = -1; mu_track = None
    current_file = rec.path
    t_start = time.time()
    n = 0
    for ev in rec.physics_events(max_events=args.max_events):
        n += 1
        h = decode_header(ev["Header"])
        ts = int(h.timestamp)
        r = rec.reconstruct(ev, fit_vertex=False)
        hits = r.hits
        mh = muon_hits(hits)
        idm = mh["cable"] < N_ID
        q17 = float(mh["q"][mh["cable"] < N_ID17].sum()); q20 = float(mh["q"][idm & (mh["cable"] >= N_ID17)].sum())
        q_od = float(mh["q"][~idm].sum())
        n_od = n200_od(mh["t"][~idm])
        muon = is_muon(q17, n_od)
        row = {
            "run": h.run, "file": file_numbers.get(getattr(rec, "current_path", rec.path), -1), "index": ev.index, "event": h.event_number,
            "unix_time": h.time_s, "timestamp": ts, "trigger": h.trigger_type & 0xFFFFFFFF,
            "nsum": h.nsum, "nsum_max": h.nsum_max,
            "nhit": int(idm.sum()), "nhit_od": int((~idm).sum()), "nsat": int(mh["saturated"][idm].sum()), "nwave": len(hits),
            "q17": q17, "q20": q20, "q_od": q_od, "q_total": q17 + q20,
            "is_muon": muon, "retrigger": False, "od_only": False, "n200_od": n_od,
            "dt_prev_us": (ts - prev_ts) * TICK_US if prev_ts is not None else np.nan,
            "dt_muon_us": (ts - mu_ts) * TICK_US if mu_ts is not None else np.nan, "prev_muon": mu_event,
            "dist_track": np.nan,
            "x": np.nan, "y": np.nan, "z": np.nan, "t0": np.nan, "r": np.nan, "sigma_t": np.nan, "n_used": 0,
            "vertex_ok": False, "e_hit": np.nan, "e_charge": np.nan, "q_window": np.nan, "n_window": 0,
            "mu_converged": False, "mu_chimney": False,
        }
        for c in MU_COLS:
            row[c] = np.nan
        if muon:
            retrig = mu_ts is not None and (ts - mu_ts) < retrig_ticks and q17 < 10000.0
            od_only = q17 < 10000.0 and int(idm.sum()) < 500
            row["retrigger"] = retrig; row["od_only"] = od_only
            if not retrig and (not od_only or args.fit_od_muons):
                od = mh[~idm]
                q5 = float(od["q"][(od["cable"] >= 2120) & (od["cable"] <= 2125)].sum())
                track = mufit.fit(mh["cable"][idm], mh["t"][idm], mh["q"][idm], q17=q17,
                                  cable_od=od["cable"], t_od=od["t"], q_5inch=q5)
                if np.isfinite(track.t0):
                    e, x, u = track.entrance, track.exit, track.direction
                    row.update({"mu_ex": e[0], "mu_ey": e[1], "mu_ez": e[2], "mu_xx": x[0], "mu_xy": x[1], "mu_xz": x[2],
                                "mu_ux": u[0], "mu_uy": u[1], "mu_uz": u[2], "mu_t0": track.t0,
                                "mu_cos_zenith": track.cos_zenith, "mu_impact": track.impact,
                                "mu_l_ls": track.l_ls, "mu_l_bo": track.l_bo, "mu_sigma_t": track.sigma_t,
                                "mu_frac_used": track.frac_used, "mu_n_used": track.n_used,
                                "mu_delta_q": q17 - (DQDX_LS * track.l_ls + DQDX_BO * track.l_bo),
                                "mu_charge_scale": track.charge_scale, "mu_converged": bool(track.converged),
                                "mu_chimney": bool(track.chimney)})
                    mu_track = track if track.converged else None
                else:
                    mu_track = None
            if not retrig:
                mu_ts = ts; mu_event = h.event_number
                if od_only and not args.fit_od_muons:
                    mu_track = None
        else:
            vertex, energy = vertex_energy(rec, hits)
            if vertex is not None and np.isfinite(vertex.x):
                row.update({"x": vertex.x, "y": vertex.y, "z": vertex.z, "t0": vertex.t0,
                            "r": float(np.linalg.norm(vertex.xyz)), "sigma_t": vertex.sigma_t,
                            "n_used": vertex.n_used, "vertex_ok": bool(vertex.ok)})
                if energy is not None:
                    row.update({"e_hit": energy.e_hit * scale["e_hit"], "e_charge": energy.e_charge * scale["e_charge"],
                                "q_window": energy.q_window, "n_window": energy.n_window})
                if mu_track is not None and mu_ts is not None:
                    row["dist_track"] = float(mu_track.distance(np.asarray(vertex.xyz, dtype=float))[0])
        rows.append(row)
        prev_ts = ts
        if not args.quiet and n % 1000 == 0:
            print(f"  {n} events, {time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    for c in ("is_muon", "retrigger", "od_only", "vertex_ok", "mu_converged", "mu_chimney"):
        df[c] = df[c].astype(bool)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df.attrs = {"files": files, "calib": args.calib, "ml_pdf": args.ml_pdf}
    df.to_parquet(args.output, index=False)
    mu = df.is_muon & ~df.retrigger
    print(f"{len(df)} physics events from {len(files)} file(s): {int(mu.sum())} muons "
          f"({int((mu & df.mu_converged).sum())} converged tracks, {int((mu & (df.mu_l_ls > 0)).sum())} through the LS), "
          f"{int(df.vertex_ok.sum())} vertices  [{time.time() - t_start:.0f} s] -> {args.output}")


if __name__ == "__main__":
    main()
