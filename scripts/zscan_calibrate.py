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
from claudland.zscan import (load_cache, calibrate_center, calibrate_light_yield, fit_velocities, build_time_pdf,
                           residuals_fixed_vertex)


def source_window(cache, frac=0.75):
    """(lo, hi) nhit window around the source peak of a cache (upper part of the nhit distribution)."""
    nh = cache["events"]["nhit"]
    med = np.median(nh[nh > 0.5 * np.percentile(nh, 90)])
    return int(frac * med), int(1.25 * med)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--center", type=int, default=2283)
    ap.add_argument("--calib-out", default=str(CFG.tq), help="output TQ calibration (default: 'tq' of claudland.toml)")
    ap.add_argument("--pdf-out", default=str(CFG.time_pdf), help="output time PDFs (default: 'time_pdf' of claudland.toml)")
    ap.add_argument("--v-ls", type=float, default=17.6, help="starting light speed (cm/ns)")
    ap.add_argument("--single-speed", action="store_true", help="force v_bo = v_ls")
    ap.add_argument("--max-z", type=float, default=550.0, help="use runs with |z| <= this (cm) for velocities/PDF")
    ap.add_argument("--t0-stat", default="mode", choices=["mode", "median"])
    ap.add_argument("--center-only", action="store_true",
                    help="use ONLY the centre run: fixed light speeds (--v-ls/--v-bo), time PDFs without distance bins")
    ap.add_argument("--v-bo", type=float, default=None, help="buffer-oil light speed for --center-only (default = v_ls)")
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
    center = caches[args.center]
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
    else:
        scan = [r for r in sorted(caches) if abs(zs[r]) <= args.max_z]
        print(f"velocity fit on runs {scan}")
        v_ls, v_bo, hist = fit_velocities([caches[r] for r in scan], [zs[r] for r in scan], calib, pmts,
                                          v_ls=args.v_ls, v_bo=args.v_ls, two_media=not args.single_speed, verbose=True)
        print(f"effective light speeds: scintillator {v_ls:.3f} cm/ns, buffer oil {v_bo:.3f} cm/ns")
        calib.meta.update({"v_ls": float(v_ls), "v_bo": float(v_bo), "center_run": args.center,
                           "velocity_runs": scan})
    # light yield per tube from the centre run
    ly = calibrate_light_yield(center, calib, pmts, v_ls, v_bo, source_like=win)
    eta = ly["eta"]; live = ly["live"]
    print(f"light yield from {ly['n_events']} centre events: {live.sum()} live tubes; eta 17\" mean {eta[:1325][live[:1325]].mean():.3f}, "
          f"20\" mean {eta[1325:1879][live[1325:1879]].mean():.3f} p.e./MeV; dark hits/window mean {ly['dark'][live].mean():.4f}; "
          f"sum eta {eta.sum():.0f} p.e./MeV")
    # time PDF
    d_edges = np.array([0.0, np.inf]) if args.center_only else None
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
