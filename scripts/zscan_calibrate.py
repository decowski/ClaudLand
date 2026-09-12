#!/usr/bin/env python3
"""Derive the detector calibration from the ⁶⁰Co z-scan hit caches.

1. T0 and Q0 per channel from the centre run (source at the origin).
2. Effective light speeds in scintillator and buffer oil from the whole scan.
3. Empirical hit-time residual densities (tube type x charge x distance).

    python scripts/zscan_calibrate.py cache/hits_*.npz --center 2283 \\
           --calib-out cache/calib_center.json --pdf-out cache/timepdf.npz
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

from claudland.calib import TQCalibration
from claudland.geometry import PMTTable
from claudland.energy import source_energy_mev
from claudland.zscan import (load_cache, calibrate_center, calibrate_light_yield, fit_velocities, build_time_pdf,
                           residuals_fixed_vertex, source_window as _source_window, select_source_events, filter_cache)


def source_window(cache):
    """(lo, hi) nhit window around the source peak of a cache (highest significant peak of the spectrum)."""
    return _source_window(cache["events"]["nhit"])


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--center", type=int, default=2283)
    ap.add_argument("--calib-out", default=str(CFG.tq), help="output TQ calibration (default: 'tq' of claudland.toml)")
    ap.add_argument("--pdf-out", default=str(CFG.time_pdf), help="output time PDFs (default: 'time_pdf' of claudland.toml)")
    ap.add_argument("--v-ls", type=float, default=17.6, help="starting light speed (cm/ns)")
    ap.add_argument("--single-speed", action="store_true", help="force v_bo = v_ls")
    ap.add_argument("--fix-v", type=float, nargs=2, metavar=("V_LS", "V_BO"), default=None,
                    help="skip the velocity fit and use these light speeds (cm/ns), e.g. values tuned on the scan with "
                         "zscan_evaluate.py; the time PDFs are still built from the whole scan")
    ap.add_argument("--max-z", type=float, default=550.0, help="use runs with |z| <= this (cm) for velocities/PDF")
    ap.add_argument("--t0-stat", default="mode", choices=["mode", "median"])
    ap.add_argument("--center-only", action="store_true",
                    help="use ONLY the centre run: fixed light speeds (--v-ls/--v-bo), time PDFs without distance bins")
    ap.add_argument("--v-bo", type=float, default=None, help="buffer-oil light speed for --center-only (default = v_ls)")
    ap.add_argument("--energy", type=float, default=None,
                    help="source energy in MeV for the light yield (default: from the run type of the centre cache and --energy-unit)")
    ap.add_argument("--energy-unit", choices=["visible", "real"], default="visible",
                    help="visible energy of the KamLAND E_vis/E_real tables (default; 60Co 2.343, 68Ge 0.846 MeV) or real gamma energy")
    ap.add_argument("--d-edges", default=None,
                    help="comma-separated distance-bin edges (cm) of the time PDFs, e.g. 0,300,450,600,750,900,1050,inf "
                         "(default: 0,400,650,900,inf)")
    ap.add_argument("--max-dist", type=float, default=150.0,
                    help="keep only events whose window-fitter vertex is within this distance (cm) of the source "
                         "(in addition to the nhit window); 0 disables the position cut")
    ap.add_argument("--radius-scale", type=float, default=1.0,
                    help="scale of the inner-PMT radius, e.g. 0.976 (=830/850) for the photocathode position")
    args = ap.parse_args()

    files = sorted(set(sum([glob.glob(f) for f in args.caches], [])))
    caches = {}
    for f in files:
        c = load_cache(f)
        caches[int(c["run"])] = c
    print(f"{len(caches)} caches: runs {sorted(caches)}")
    zs = {r: float(c["source_z_cm"]) for r, c in caches.items()}
    pmts = PMTTable.load(radius_scale=args.radius_scale)
    calib = TQCalibration()
    calib.meta["radius_scale"] = args.radius_scale
    # source selection: nhit window around the source peak + vertex near the known source position
    if args.max_dist > 0:
        t_sel = time.time()
        for r in sorted(caches):
            c = caches[r]
            sel = select_source_events(c, TQCalibration(), pmts, (0.0, 0.0, zs[r]), v_ls=args.v_ls, max_dist=args.max_dist)
            n_win = int(sel["in_window"].sum()); n_keep = int(sel["keep"].sum())
            print(f"  run {r} z={zs[r]:+5.0f}: nhit window {_source_window(c['events']['nhit'])}, {n_win} events in window, "
                  f"{n_keep} within {args.max_dist:.0f} cm of the source ({100.0 * n_keep / max(n_win, 1):.0f}%)")
            caches[r] = filter_cache(c, sel["keep"])
        print(f"source selection done [{time.time() - t_sel:.0f} s]")
    center = caches[args.center]
    energy = args.energy if args.energy is not None else source_energy_mev(str(center.get("run_type", "")), unit=args.energy_unit)
    calib.meta["source_energy_mev"] = float(energy)
    calib.meta["energy_unit"] = args.energy_unit if args.energy is None else "user"
    calib.meta["source_run_type"] = str(center.get("run_type", ""))
    print(f"source energy for the light-yield calibration: {energy:.4f} MeV")
    calib.bin_ns[:] = center["bin_ns"]          # sampling periods from the centre run's clock events
    t0 = time.time()
    win = source_window(center)
    diag = calibrate_center(center, calib, pmts, v_ls=args.v_ls, statistic=args.t0_stat, source_like=win)
    t0tab = diag["t0"]
    print(f"centre run {args.center} (z={zs[args.center]:.0f}), source window nhit {win}: "
          f"T0 for {diag['n_t0']} channels (rms {t0tab[t0tab != 0].std():.2f} ns, range {t0tab.min():.1f}..{t0tab.max():.1f}), "
          f"Q0 for {diag['n_q0']} channels (mean {calib.q1pe[:1879].mean():.1f} ADC)  [{time.time() - t0:.0f} s]")
    # residual width at the centre after calibration
    rr = residuals_fixed_vertex(center, calib, pmts, (0, 0, 0), args.v_ls)
    res17 = rr["res"][rr["cable"] < 1325]
    print(f"  centre residual (17\"): mode {np.median(res17[np.abs(res17) < 10]):.2f}, FWHM-ish robust sigma "
          f"{1.4826 * np.median(np.abs(res17 - np.median(res17))):.2f} ns")
    if args.center_only:
        v_ls = args.v_ls; v_bo = args.v_bo if args.v_bo is not None else args.v_ls
        scan = [args.center]
        print(f"centre-only calibration: light speeds fixed at {v_ls:.2f} / {v_bo:.2f} cm/ns (external input)")
        calib.meta.update({"v_ls": float(v_ls), "v_bo": float(v_bo), "center_run": args.center, "velocity_runs": []})
    elif args.fix_v is not None:
        v_ls, v_bo = args.fix_v
        scan = [r for r in sorted(caches) if abs(zs[r]) <= args.max_z]
        print(f"light speeds fixed at {v_ls:.2f} / {v_bo:.2f} cm/ns (external / tuned input); PDFs from runs {scan}")
        calib.meta.update({"v_ls": float(v_ls), "v_bo": float(v_bo), "center_run": args.center, "velocity_runs": [],
                           "v_source": "fixed (--fix-v)"})
    else:
        scan = [r for r in sorted(caches) if abs(zs[r]) <= args.max_z]
        print(f"velocity fit on runs {scan}")
        v_ls, v_bo, hist = fit_velocities([caches[r] for r in scan], [zs[r] for r in scan], calib, pmts,
                                          v_ls=args.v_ls, v_bo=args.v_ls, two_media=not args.single_speed, verbose=True)
        print(f"effective light speeds: scintillator {v_ls:.3f} cm/ns, buffer oil {v_bo:.3f} cm/ns")
        calib.meta.update({"v_ls": float(v_ls), "v_bo": float(v_bo), "center_run": args.center,
                           "velocity_runs": scan})
    # light yield per tube from the centre run
    ly = calibrate_light_yield(center, calib, pmts, v_ls, v_bo, source_like=win, energy_mev=energy)
    eta = ly["eta"]; live = ly["live"]
    print(f"light yield from {ly['n_events']} centre events: {live.sum()} live tubes; eta 17\" mean {eta[:1325][live[:1325]].mean():.3f}, "
          f"20\" mean {eta[1325:1879][live[1325:1879]].mean():.3f} p.e./MeV; dark hits/window mean {ly['dark'][live].mean():.4f}; "
          f"sum eta {eta.sum():.0f} p.e./MeV")
    # time PDF
    d_edges = np.array([0.0, np.inf]) if args.center_only else None
    if args.d_edges and not args.center_only:
        d_edges = np.array([float(x) for x in args.d_edges.split(",")])
        calib.meta["pdf_d_edges"] = [float(x) for x in d_edges]
    kw = {"d_edges": d_edges} if d_edges is not None else {}
    pdf = build_time_pdf([caches[r] for r in scan], [zs[r] for r in scan], calib, pmts, v_ls, v_bo, **kw)
    cnt = pdf.counts.sum(-1)
    print("time-PDF hit counts per (type, q-bin, d-bin):")
    for ti, name in enumerate(("17\"", "20\"")):
        for qi in range(cnt.shape[1]):
            print(f"  {name} q-bin {qi}: " + " ".join(f"{int(v):8d}" for v in cnt[ti, qi]))
    calib.save(args.calib_out)
    pdf.save(args.pdf_out)
    print(f"written {args.calib_out}, {args.pdf_out}  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
