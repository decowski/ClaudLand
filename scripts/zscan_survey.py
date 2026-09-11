#!/usr/bin/env python3
"""Geometry survey: per-tube position and time offsets from the z-scan residuals.

Fits dP_i (cm) and dT_i (ns) for every inner tube from the run-by-run modes of
its time residual with the vertex fixed at the known source position.  This is
a diagnostic of the mechanical tolerances (sphere shape, bolt pattern), so it
necessarily uses the whole scan.  With ``--eval`` the likelihood fitter is
rerun with the surveyed positions/offsets applied to quantify their effect.

    python scripts/zscan_survey.py cache/hits_*.npz --calib cache/calib_center.json -o cache/survey
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from claudland import config as _config

CFG = _config.get()

import numpy as np

from claudland.calib import TQCalibration
from claudland.geometry import PMTTable, N_ID, N_ID17
from claudland.zscan import load_cache, survey_positions, deformation_vs_theta


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--max-z", type=float, default=550.0)
    ap.add_argument("--prior", type=float, default=30.0, help="Gaussian prior on |dP| (cm)")
    ap.add_argument("-o", "--output", default=str(CFG.cache_dir / "survey"))
    ap.add_argument("--v-ls", type=float, default=None, help="override the light speed (degeneracy check)")
    args = ap.parse_args()
    files = sorted(set(sum([glob.glob(f) for f in args.caches], [])))
    calib = TQCalibration.load(args.calib)
    v_ls = float(calib.meta.get("v_ls", 19.4)); v_bo = float(calib.meta.get("v_bo", v_ls))
    if args.v_ls is not None:
        v_ls = v_bo = args.v_ls
    pmts = PMTTable.load(radius_scale=float(calib.meta.get("radius_scale", 1.0)))
    caches, zs = [], []
    for f in files:
        c = load_cache(f)
        if abs(float(c["source_z_cm"])) <= args.max_z:
            caches.append(c); zs.append(float(c["source_z_cm"]))
    print(f"survey from {len(caches)} runs, |z| <= {args.max_z:.0f} cm, light speeds {v_ls:.2f}/{v_bo:.2f}")
    sv = survey_positions(caches, zs, calib, pmts, v_ls, v_bo, sigma_prior_cm=args.prior)
    dP, dT, err, nrun = sv["dP"], sv["dT"], sv["err"], sv["nrun"]
    good = (nrun >= 15) & np.isfinite(err[:, 3])
    P = pmts.xyz[:N_ID]; r = np.linalg.norm(P, axis=1); rhat = P / r[:, None]
    dr = np.einsum("ij,ij->i", dP, rhat)
    # tangential components
    zhat = np.array([0, 0, 1.0]); th = np.cross(rhat, np.cross(zhat, rhat)); th /= np.maximum(np.linalg.norm(th, axis=1), 1e-9)[:, None]
    ph = np.cross(zhat[None, :], rhat); ph /= np.maximum(np.linalg.norm(ph, axis=1), 1e-9)[:, None]
    dth = np.einsum("ij,ij->i", dP, th); dph = np.einsum("ij,ij->i", dP, ph)
    print(f"{good.sum()} tubes with >= 15 runs; per-run fit residual rms (median over tubes) {np.nanmedian(sv['rms'][good]):.2f} ns")
    for name, m in (("17\"", good & (np.arange(N_ID) < N_ID17)), ("20\"", good & (np.arange(N_ID) >= N_ID17))):
        print(f"  {name}: dr  mean {dr[m].mean():+6.1f} rms {dr[m].std():5.1f} cm (median err {np.nanmedian(err[m, 1:4].mean(1)):.1f});  "
              f"d_theta rms {dth[m].std():5.1f}  d_phi rms {dph[m].std():5.1f} cm;  dT rms {dT[m].std():.2f} ns")
    # statistical expectation: compare rms with median error
    print(f"  median fitted uncertainty per component: {np.nanmedian(err[good, 1]):.1f}, {np.nanmedian(err[good, 2]):.1f}, {np.nanmedian(err[good, 3]):.1f} cm -> "
          f"excess (true scatter) dr ~ {np.sqrt(max(dr[good].var() - np.nanmedian(err[good, 1:4].mean(1)) ** 2, 0)):.1f} cm")
    rows, _ = deformation_vs_theta(pmts, dP, good=good)
    print("\nradial deviation vs cos(theta) (theta from +z):")
    print(f"{'cos':>6s} {'n':>5s} {'<dr>':>7s} {'err':>5s} {'<dz>':>7s} {'<|dxy|>':>8s}")
    for row in rows:
        print(f"{row[0]:6.2f} {int(row[1]):5d} {row[2]:+7.1f} {row[3]:5.1f} {row[4]:+7.1f} {row[5]:8.1f}")
    np.savez(args.output + ".npz", dP=dP, dT=dT, err=err, nrun=nrun, rms=sv["rms"], good=good, zs=np.array(zs))
    print("saved", args.output + ".npz")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        cost = P[:, 2] / r; phi = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
        sc = axs[0].scatter(phi[good], np.degrees(np.arccos(cost[good])), c=dr[good], cmap="coolwarm", vmin=-30, vmax=30, s=8)
        plt.colorbar(sc, ax=axs[0], label="radial offset [cm]"); axs[0].set_xlabel("phi [deg]"); axs[0].set_ylabel("theta [deg]"); axs[0].invert_yaxis()
        axs[1].errorbar(rows[:, 0], rows[:, 2], rows[:, 3], fmt="o-"); axs[1].axhline(0, color="k", lw=0.5)
        axs[1].set_xlabel("cos theta"); axs[1].set_ylabel("<radial offset> [cm]")
        axs[2].hist(dr[good], bins=60, range=(-60, 60), histtype="step", label="radial")
        axs[2].hist(dth[good], bins=60, range=(-60, 60), histtype="step", label="theta")
        axs[2].hist(dph[good], bins=60, range=(-60, 60), histtype="step", label="phi")
        axs[2].set_xlabel("offset [cm]"); axs[2].legend()
        fig.suptitle("PMT position survey from the 60Co z-scan (time residual modes, vertex fixed at source)")
        fig.tight_layout(); fig.savefig(args.output + ".png", dpi=110); print("plot", args.output + ".png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
