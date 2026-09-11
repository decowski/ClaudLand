#!/usr/bin/env python3
"""Select muons in a run file and fit their tracks.

Example
-------
    python3 scripts/muon_fit.py ../Run/run_001467_000000_000001.sf --calib cache/calib_center.json \
            -o cache/muons_1467.npz --keep-hits

Output ``.npz``: ``muons`` (MUON_DTYPE table); with ``--keep-hits`` also ``hits``
(MUHIT_DTYPE, all selected events concatenated), ``hit_event`` (row index per hit)
and ``residual`` (ns, NaN for OD tubes).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from claudland import config as _config

CFG = _config.get()
from claudland.reco import EventReconstructor
from claudland.banks import decode_header, decode_hit_header
from claudland.geometry import PMTTable, N_ID, N_ID17
from claudland.calib import TQCalibration
from claudland.trigger import OD_TOP, OD_UPPER, OD_LOWER, OD_BOTTOM

OD_TRIGGER_BITS = OD_TOP | OD_UPPER | OD_LOWER | OD_BOTTOM
from claudland.muon import MuonTrackFitter, MuonChargeModel, muon_hits, is_muon, n200_od, muon_row, MUON_DTYPE, MUHIT_DTYPE


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-o", "--output")
    ap.add_argument("--calib", default=str(CFG.optional("tq") or ""),
                    help="JSON calibration (bin widths, Q0, T0); default: 'tq' of claudland.toml if present")
    ap.add_argument("-n", "--max-events", type=int, default=None, help="physics events to scan")
    ap.add_argument("--min-nhit", type=int, default=200, help="HitHeader multiplicity pre-selection")
    ap.add_argument("--index", default="1.5", help='effective refractive index: a number (default 1.5), "kat" or "fit"')
    ap.add_argument("--time-only", action="store_true", help="no per-tube charge model in the fit")
    ap.add_argument("--no-od", action="store_true", help="do not use the OD hit times in the track scoring")
    ap.add_argument("--charge-weight", type=float, default=10.0, help="weight of the charge term in the joint fit")
    ap.add_argument("--weight", default="uniform", choices=["uniform", "charge", "kat"])
    ap.add_argument("--seed", default="scan", choices=["scan", "cluster", "both"], help="track seeding: global grid scan (default) or Kat cluster heuristics")
    ap.add_argument("--n-scan", type=int, default=120, help="grid points on the sphere for the scan")
    ap.add_argument("--time-key", default="t", choices=["t", "t_lead"],
                    help="t = T0-corrected CFD time; t_lead = leading-edge sample time (no T0)")
    ap.add_argument("--all", action="store_true", help="fit every pre-selected event, not only Kat-selected muons")
    ap.add_argument("--od-trigger", action="store_true",
                    help="require an outer-detector trigger bit (0x00F00000) for the low-charge branch of the selection "
                         "(needed in calibration-source runs, where source events reach N200_OD = 5-7)")
    ap.add_argument("--keep-hits", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    index = args.index
    if index not in ("kat", "fit"):
        index = float(index)
    pmts = PMTTable.load()
    calib = TQCalibration.load(args.calib) if args.calib else None
    rec = EventReconstructor(args.file, pmts=pmts, calib=calib, gains=(0, 1, 2), verbose=not args.quiet)
    rec.prepare()
    live17 = rec.energy_estimator.live[:N_ID17] if rec.energy_estimator.live is not None else None
    cm = None if args.time_only else MuonChargeModel(pmts, index=index if isinstance(index, float) else 1.5, live=live17)
    fitter = MuonTrackFitter(pmts, index=index, charge_model=cm, charge_weight=args.charge_weight,
                             weight=args.weight, seed=args.seed, n_scan=args.n_scan)

    rows, hits_all, hit_ev, resid_all = [], [], [], []
    n_scanned = n_pre = 0
    t_start = time.time()
    for ev in rec.physics_events(max_events=args.max_events):
        n_scanned += 1
        hh = decode_hit_header(ev["HitHeader"])
        if hh.nhit < args.min_nhit:
            continue
        n_pre += 1
        r = rec.reconstruct(ev, fit_vertex=False)
        mh = muon_hits(r.hits, time_key=args.time_key)
        idm = mh["cable"] < N_ID
        q17 = float(mh["q"][mh["cable"] < N_ID17].sum())
        n_od = n200_od(mh["t"][~idm])
        if not (args.all or is_muon(q17, n_od)):
            continue
        if args.od_trigger and q17 < 10000 and not (r.header.trigger_type & OD_TRIGGER_BITS):
            continue
        od = mh[~idm]
        q5 = float(od["q"][(od["cable"] >= 2120) & (od["cable"] <= 2125)].sum())
        track = fitter.fit(mh["cable"][idm], mh["t"][idm], mh["q"][idm], q17=q17,
                           cable_od=None if args.no_od else od["cable"], t_od=None if args.no_od else od["t"], q_5inch=q5)
        row = muon_row(r.header, ev.index, mh, track, pmts)
        rows.append(row)
        if not args.quiet:
            print(f"ev {ev.index:6d} nhit={row['nhit']:4d} N200OD={n_od:3d} Q17={q17:9.0f} Q20={row['q20']:8.0f} "
                  f"QOD={row['q_od']:7.0f} | {track.summary()}", flush=True)
        if args.keep_hits:
            hits_all.append(mh)
            hit_ev.append(np.full(len(mh), len(rows) - 1, dtype=np.int32))
            res = np.full(len(mh), np.nan, dtype=np.float32)
            if np.isfinite(track.t0):
                res[idm] = fitter.residuals(track, mh["cable"][idm], mh["t"][idm])
            resid_all.append(res)
    dt = time.time() - t_start
    tab = np.array(rows, dtype=MUON_DTYPE) if rows else np.zeros(0, dtype=MUON_DTYPE)
    print(f"{n_scanned} physics events scanned, {n_pre} with nhit >= {args.min_nhit}, {len(tab)} muons fitted in {dt:.1f} s")
    if len(tab):
        conv = tab["converged"]
        ls = tab["l_ls"] > 0
        print(f"  converged {conv.sum()}/{len(tab)}; LS-crossing {ls.sum()}; median sigma_t {np.median(tab['sigma_t'][conv]):.2f} ns; "
              f"median frac_used {np.median(tab['frac_used'][conv]):.2f}")
        if (conv & ls).any():
            dq = tab["q17"][conv & ls] / tab["l_ls"][conv & ls]
            print(f"  Q17 / L_LS: median {np.median(dq):.0f} p.e./cm (LS muons)")
        bo = conv & ~ls & (tab["l_bo"] > 100)
        if bo.any():
            print(f"  Q17 / L_BO: median {np.median(tab['q17'][bo] / tab['l_bo'][bo]):.0f} p.e./cm (buffer-oil muons)")
    if args.output:
        out = dict(muons=tab, run=tab["run"][0] if len(tab) else -1)
        if args.keep_hits and hits_all:
            out["hits"] = np.concatenate(hits_all); out["hit_event"] = np.concatenate(hit_ev)
            out["residual"] = np.concatenate(resid_all)
        np.savez_compressed(args.output, **out)
        print(f"written {args.output}")


if __name__ == "__main__":
    main()
