#!/usr/bin/env python3
"""Reconstruct the physics events of one KamLAND run file.

Steps: pedestal/clock calibration from the run's own calibration events, TQ
extraction, vertex fit, energy estimate.  Optionally derives per-channel time
offsets (T0) from the run itself before the final pass.

Examples::

    # full run, default constants, write an .npz table
    python scripts/reco_run.py run_002279_000000_000001.sfz -o reco_2279.npz

    # derive T0 from the known source position (RunHeader comment '+3.50 m' -> z = 350 cm),
    # save the calibration, then reconstruct
    python scripts/reco_run.py run_002279_000000_000001.sfz --t0-source --save-calib calib_2279.json -o reco_2279.npz

    # apply an existing calibration to another run
    python scripts/reco_run.py run_002278_000000_000001.sfz --calib calib_2279.json -o reco_2278.npz
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from claudland import config as _config

CFG = _config.get()

import numpy as np

from claudland.reco import EventReconstructor, EVENT_DTYPE
from claudland.calib import TQCalibration
from claudland.vertex import VertexFitter
from claudland.energy import EnergyEstimator, CO60_ENERGY_MEV, source_energy_mev
from claudland.geometry import PMTTable
from claudland.zscan import source_peak_nhit


def summarize(tab: np.ndarray, source_z=None) -> str:
    """Print a summary of a reconstructed-event table."""
    lines = []
    ok = tab["vertex_ok"]
    lines.append(f"{len(tab)} physics events, {ok.sum()} with a converged vertex")
    if ok.sum() == 0:
        return "\n".join(lines)
    peak = source_peak_nhit(tab["nhit"][ok])
    src = ok & (tab["nhit"] > 0.8 * peak) & (tab["nhit"] < 1.2 * peak)
    for label, m in (("all fitted", ok), ("source-like (nhit peak)", src)):
        z = tab["z"][m]; x = tab["x"][m]; y = tab["y"][m]
        rz = 1.4826 * np.median(np.abs(z - np.median(z)))
        lines.append(f"  {label:24s}: n={m.sum():5d} nhit={np.median(tab['nhit'][m]):.0f} "
                     f"x={np.median(x):7.1f} y={np.median(y):7.1f} z={np.median(z):7.1f} cm (robust sigma_z {rz:.1f}) "
                     f"sigma_t={np.median(tab['sigma_t'][m]):.2f} ns  E_charge={np.median(tab['e_charge'][m]):.3f} "
                     f"E_hit={np.median(tab['e_hit'][m]):.3f} MeV  Q_window={np.median(tab['q_window'][m]):.0f} p.e.")
    if source_z is not None:
        z = tab["z"][src]
        lines.append(f"  source z bias: {np.median(z) - source_z:+.1f} cm (true z = {source_z:.0f} cm)")
    return "\n".join(lines)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="the run's files in order (only the first has the calibration block)")
    ap.add_argument("-o", "--output", help="output .npz (event table, optionally hits)")
    ap.add_argument("--csv", help="also write the event table as CSV")
    ap.add_argument("-n", "--max-events", type=int, default=None, help="stop after this many physics events")
    ap.add_argument("--calib", default=str(CFG.optional("tq") or ""),
                    help="JSON calibration (bin widths, q1pe, t0) to apply; default: 'tq' of claudland.toml if present")
    ap.add_argument("--save-calib", help="write the calibration used (after any T0/Q0 derivation) to JSON")
    ap.add_argument("--t0-source", action="store_true",
                    help="derive per-channel T0 from the calibration-source position given in the RunHeader")
    ap.add_argument("--source-z", type=float, help="override the source z position (cm) for --t0-source")
    ap.add_argument("--t0-self", action="store_true", help="derive per-channel T0 from the fitted vertices")
    ap.add_argument("--q0", action="store_true", help="derive per-channel 1 p.e. charges from the hit charge spectra")
    ap.add_argument("--calib-events", type=int, default=1500, help="physics events used for T0/Q0 derivation")
    ap.add_argument("--energy-scale", action="store_true",
                    help="set the energy scale so the source-like events peak at the source energy")
    ap.add_argument("--source-energy", type=float, default=None,
                    help="source energy (MeV) for --energy-scale; default: from the RunHeader run type and --energy-unit")
    ap.add_argument("--energy-unit", choices=["visible", "real"], default="visible",
                    help="scale to the source's visible energy of the KamLAND E_vis/E_real tables (default; 60Co 2.343, "
                         "68Ge 0.846 MeV) or to its real gamma energy (60Co 2.506, 68Ge 1.022 MeV)")
    ap.add_argument("--v-ls", type=float, default=None,
                    help="effective light speed in the scintillator (cm/ns); default: from the calibration, else 17.6")
    ap.add_argument("--v-bo", type=float, default=None, help="effective light speed in the buffer oil (default: from calibration or = v_ls)")
    ap.add_argument("--ml-pdf", help="time-PDF file (from zscan_calibrate.py): use the maximum-likelihood vertex fitter; "
                                     "'auto' takes 'time_pdf' of claudland.toml")
    ap.add_argument("--no-t0", action="store_true", help="ignore the per-channel time offsets of the loaded calibration")
    ap.add_argument("--joint", choices=["hit", "poisson"], default=None,
                    help="with --ml-pdf: joint time + charge likelihood (hit pattern or Poisson charge), fits the energy too")
    ap.add_argument("--keep-hits", action="store_true", help="store per-hit arrays in the .npz")
    ap.add_argument("--time-key", default="t_cfd", choices=["t_cfd", "t_lead", "t_peak"])
    ap.add_argument("--all-gains", action="store_true", help="decode medium/low gain waveforms too")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if args.ml_pdf == "auto":
        args.ml_pdf = str(CFG.require("time_pdf"))
    calib = TQCalibration.load(args.calib) if args.calib else None
    if calib is not None and args.no_t0:
        calib.t0[:] = 0.0
    pmts = PMTTable.load()
    v_ls = args.v_ls or (float(calib.meta["v_ls"]) if calib and "v_ls" in calib.meta else 17.6)
    v_bo = args.v_bo or (float(calib.meta["v_bo"]) if calib and "v_bo" in calib.meta else None)
    fitter = VertexFitter(pmts, v_ls=v_ls, v_bo=v_bo)
    energy_model = EnergyEstimator(pmts, eta=getattr(calib, "eta", None), eta_charge=getattr(calib, "eta_q", None),
                                   dark_per_tube=getattr(calib, "dark", None))
    if calib is not None and calib.eta is not None:
        energy_model.live = calib.eta[:1879] > 0
    if args.ml_pdf:
        from claudland.vertex_ml import MLVertexFitter, ChargeTimeVertexFitter
        from claudland.zscan import TimePDF
        pdf = TimePDF.load(args.ml_pdf)
        if args.joint:
            fitter = ChargeTimeVertexFitter(pmts, pdf, v_ls, v_bo, energy_model=energy_model, charge_model=args.joint, prefit=fitter)
        else:
            fitter = MLVertexFitter(pmts, pdf, v_ls, v_bo, prefit=fitter)
    rec = EventReconstructor(args.files, pmts=pmts, calib=calib, vertex=fitter, energy=energy_model,
                             gains=(0, 1, 2) if args.all_gains else (0,), time_key=args.time_key,
                             verbose=not args.quiet)
    rec.prepare()
    source_z = args.source_z
    if source_z is None and rec.run_header is not None:
        source_z = rec.run_header.source_z_cm

    if args.t0_source or args.t0_self or args.q0:
        t = time.time()
        rec.run(max_events=args.calib_events, keep_hits=True, progress_every=0)
        hits = rec.hit_arrays
        if args.q0:
            rec.calibrate_q1pe(hits)
            rec.log(f"Q0 calibration: mean 1 p.e. ADC sum {rec.calib.q1pe[:1879].mean():.1f}")
        if args.t0_source:
            if source_z is None:
                sys.exit("--t0-source needs a source position (RunHeader comment or --source-z)")
            t0 = rec.calibrate_t0(hits, fixed_vertex=(0.0, 0.0, source_z))
            rec.log(f"T0 calibration from source at z={source_z:.0f} cm: rms {t0[t0 != 0].std():.2f} ns over {(t0 != 0).sum()} channels")
        elif args.t0_self:
            t0 = rec.calibrate_t0(hits)
            rec.log(f"T0 calibration from fitted vertices: rms {t0[t0 != 0].std():.2f} ns over {(t0 != 0).sum()} channels")
        rec.log(f"calibration pass: {time.time() - t:.1f} s")

    tab = rec.run(max_events=args.max_events, keep_hits=args.keep_hits)
    if not args.energy_scale and len(tab):
        # apply energy scales stored in a loaded calibration
        for key in ("e_charge", "e_hit"):
            sc = rec.calib.meta.get(f"scale_{key}")
            if sc:
                tab[key] *= float(sc)
    if args.energy_scale and len(tab):
        ok = tab["vertex_ok"]
        peak = source_peak_nhit(tab["nhit"][ok])
        src = ok & (tab["nhit"] > 0.8 * peak) & (tab["nhit"] < 1.2 * peak)
        e_src = args.source_energy
        if e_src is None:
            e_src = source_energy_mev(rec.run_header.run_type if rec.run_header else "", unit=args.energy_unit)
        rec.calib.meta["source_energy_mev"] = float(e_src)
        rec.calib.meta["energy_unit"] = args.energy_unit if args.source_energy is None else "user"
        for key in ("e_charge", "e_hit"):
            med = np.median(tab[key][src])
            if np.isfinite(med) and med > 0:
                tab[key] *= e_src / med
                rec.calib.meta[f"scale_{key}"] = float(e_src / med)
        rec.log(f"energy scale set on {src.sum()} source-like events: {rec.calib.meta}")

    print(summarize(tab, source_z))
    if args.save_calib:
        rec.calib.save(args.save_calib)
        print(f"calibration written to {args.save_calib}")
    if args.output:
        payload = {"events": tab, "run": rec.run_header.run if rec.run_header else -1,
                   "run_type": rec.run_header.run_type if rec.run_header else "",
                   "run_comment": rec.run_header.comment if rec.run_header else ""}
        if args.keep_hits:
            lengths = np.array([len(h) for h in rec.hit_arrays])
            payload["hits"] = np.concatenate(rec.hit_arrays) if len(rec.hit_arrays) else np.zeros(0)
            payload["hit_offsets"] = np.concatenate([[0], np.cumsum(lengths)])
        np.savez_compressed(args.output, **payload)
        print(f"table written to {args.output}")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(EVENT_DTYPE.names) + "\n")
            for row in tab:
                f.write(",".join(str(v) for v in row) + "\n")
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
