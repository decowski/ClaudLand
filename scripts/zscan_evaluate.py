#!/usr/bin/env python3
"""Evaluate vertex algorithms on the ⁶⁰Co z-scan: bias and resolution versus source position.

Fitters compared (all with the same T0/Q0 from the centre run):
  A  window ("push") fitter, single effective light speed
  B  window fitter, two-medium light speeds
  C  maximum-likelihood fitter with empirical time PDFs, two-medium speeds
  D  joint time + hit-pattern (Bernoulli) likelihood, energy fitted simultaneously
  E  joint time + charge (Poisson) likelihood

    python scripts/zscan_evaluate.py cache/hits_*.npz --calib cache/calib_center.json \\
           --pdf cache/timepdf.npz -n 600 -o cache/zscan_eval
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
from claudland.geometry import PMTTable, N_ID
from claudland.vertex import VertexFitter
from claudland.vertex_ml import MLVertexFitter, ChargeTimeVertexFitter
from claudland.zscan import load_cache, hit_times, hit_charges, TimePDF
from claudland.energy import EnergyEstimator

RES_DTYPE = np.dtype([("run", "i4"), ("ztrue", "f4"), ("ev", "i4"), ("nhit", "i4"), ("algo", "U1"),
                      ("x", "f4"), ("y", "f4"), ("z", "f4"), ("t0", "f4"), ("ok", "?"), ("n_used", "i4"),
                      ("sigma_t", "f4"), ("it", "i4"), ("e_hit", "f4"), ("e_charge", "f4")])


def robust_sigma(x):
    """Robust width estimate: 1.4826 times the median absolute deviation."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def tune_single_speed(files, calib, pmts, n_events=80, speeds=np.arange(16.0, 20.01, 0.5), max_z=550.0):
    """Choose the effective light speed of the window fitter that minimises the rms z bias over the scan."""
    data = []
    for f in files:
        c = load_cache(f)
        ztrue = float(c["source_z_cm"])
        if abs(ztrue) > max_z:
            continue
        nh = c["events"]["nhit"]
        peak = np.median(nh[nh > 0.5 * np.percentile(nh, 90)])
        good = np.flatnonzero((nh > 0.75 * peak) & (nh < 1.25 * peak))[:n_events]
        t = hit_times(c, calib); q = hit_charges(c, calib); h = c["hits"]
        base = h["primary"] & (h["cable"] < N_ID) & np.isfinite(h["t_cfd"])
        evs = [(h["cable"][m], t[m], q[m]) for m in (base & (h["ev"] == e) for e in good)]
        data.append((ztrue, evs))
    best = (np.inf, None)
    for v in speeds:
        fit = VertexFitter(pmts, v_ls=v)
        biases = []
        for ztrue, evs in data:
            z = np.array([fit.fit(*e).z for e in evs])
            z = z[np.isfinite(z)]
            if len(z):
                biases.append(np.median(z) - ztrue)
        rms = float(np.sqrt(np.mean(np.square(biases))))
        if rms < best[0]:
            best = (rms, v)
    return float(best[1])


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("-n", "--events", type=int, default=600, help="source-like events per run")
    ap.add_argument("--v-single", type=float, default=None,
                    help="single speed for algo A (default: tuned on the scan to minimise the rms z bias)")
    ap.add_argument("--algos", default="ACD")
    ap.add_argument("-o", "--output", default=str(CFG.cache_dir / "zscan_eval"))
    ap.add_argument("--radius-scale", type=float, default=None, help="inner-PMT radius scale (default: from calib)")
    args = ap.parse_args()

    files = sorted(set(sum([glob.glob(f) for f in args.caches], [])))
    calib = TQCalibration.load(args.calib)
    pdf = TimePDF.load(args.pdf)
    v_ls = float(calib.meta.get("v_ls", 17.6)); v_bo = float(calib.meta.get("v_bo", v_ls))
    rs = args.radius_scale if args.radius_scale is not None else float(calib.meta.get("radius_scale", 1.0))
    pmts = PMTTable.load(radius_scale=rs)
    print(f"PMT radius scale {rs:.4f}; light speeds {v_ls:.2f} / {v_bo:.2f} cm/ns")
    v_single = args.v_single
    if v_single is None and "A" in args.algos:
        v_single = tune_single_speed(files, calib, pmts)
        print(f"tuned single light speed for the window fitter: {v_single:.2f} cm/ns")
    elif v_single is None:
        v_single = v_ls
    fitA = VertexFitter(pmts, v_ls=v_single)
    fitB = VertexFitter(pmts, v_ls=v_ls, v_bo=v_bo)
    fitC = MLVertexFitter(pmts, pdf, v_ls, v_bo, prefit=fitB)
    energy = EnergyEstimator(pmts, eta=calib.eta, eta_charge=calib.eta_q, dark_per_tube=calib.dark)
    if calib.eta is not None:
        energy.live = calib.eta[:N_ID] > 0
    fitD = ChargeTimeVertexFitter(pmts, pdf, v_ls, v_bo, energy_model=energy, charge_model="hit", prefit=fitB)
    fitE = ChargeTimeVertexFitter(pmts, pdf, v_ls, v_bo, energy_model=energy, charge_model="poisson", prefit=fitB)
    fitters = {"A": fitA, "B": fitB, "C": fitC, "D": fitD, "E": fitE}
    rows = []
    t_start = time.time()
    for f in files:
        c = load_cache(f)
        run = int(c["run"]); ztrue = float(c["source_z_cm"])
        nh = c["events"]["nhit"]
        peak = np.median(nh[nh > 0.5 * np.percentile(nh, 90)])
        good = np.flatnonzero((nh > 0.75 * peak) & (nh < 1.25 * peak))[:args.events]
        t = hit_times(c, calib); q = hit_charges(c, calib)
        h = c["hits"]
        base = h["primary"] & (h["cable"] < N_ID) & np.isfinite(h["t_cfd"])
        if calib.eta is None:
            energy.live = c["live"] if "live" in c else energy.live
        by_ev = np.argsort(h["ev"], kind="stable")
        ev_sorted = h["ev"][by_ev]
        starts = np.searchsorted(ev_sorted, good, side="left"); ends = np.searchsorted(ev_sorted, good, side="right")
        n_ok = {a: 0 for a in args.algos}
        for e, a0, a1 in zip(good, starts, ends):
            idx = by_ev[a0:a1]
            idx = idx[base[idx]]
            cab, tt, qq = h["cable"][idx], t[idx], q[idx]
            res = {}
            if "A" in args.algos:
                res["A"] = fitA.fit(cab, tt, qq)
            if any(a in args.algos for a in "BCDE"):
                rB = fitB.fit(cab, tt, qq)
                st = dict(start=rB.xyz if rB.ok else None, T_start=rB.t0 if rB.ok else None)
                if "B" in args.algos:
                    res["B"] = rB
                if "C" in args.algos:
                    res["C"] = fitC.fit(cab, tt, qq, **st)
                if "D" in args.algos:
                    res["D"] = fitD.fit(cab, tt, qq, **st)
                if "E" in args.algos:
                    res["E"] = fitE.fit(cab, tt, qq, **st)
            for a, r in res.items():
                e_hit = e_ch = np.nan
                if r.ok:
                    fitter = fitters[a]
                    tau = fitter.residuals(r, cab, tt)
                    en = energy.estimate(cab, tau, qq, r.xyz)
                    e_hit, e_ch = en.e_hit, en.e_charge
                    if a in "DE":
                        e_hit = r.energy          # energy from the joint fit
                    n_ok[a] += 1
                rows.append((run, ztrue, e, int(nh[e]), a, r.x, r.y, r.z, r.t0, r.ok, r.n_used, r.sigma_t, r.iterations,
                             e_hit, e_ch))
        print(f"run {run} z={ztrue:+6.0f}: {len(good)} events, fitted " +
              ", ".join(f"{a}:{n_ok[a]}" for a in args.algos) + f"  [{time.time() - t_start:.0f} s]", flush=True)
    tab = np.array(rows, dtype=RES_DTYPE)
    np.save(args.output + ".npy", tab)
    # summary table
    print("\nbias (median z - true z) and robust sigma [cm]; rho = sqrt(x^2+y^2) median")
    hdr = f"{'run':>5s} {'z':>6s} " + " ".join(f"{a+'_bias':>7s} {a+'_sig':>6s} {a+'_rho':>6s} {a+'_sx':>5s}" for a in args.algos)
    print(hdr)
    summ = {a: [] for a in args.algos}
    for run in np.unique(tab["run"]):
        line = f"{run:5d} {tab['ztrue'][tab['run'] == run][0]:+6.0f} "
        for a in args.algos:
            m = (tab["run"] == run) & (tab["algo"] == a) & tab["ok"]
            if m.sum() < 5:
                line += f"{'-':>7s} {'-':>6s} {'-':>6s} {'-':>5s} "
                continue
            z = tab["z"][m]; x = tab["x"][m]; y = tab["y"][m]
            bias = np.median(z) - tab["ztrue"][m][0]
            summ[a].append((tab["ztrue"][m][0], bias, robust_sigma(z), np.median(np.hypot(x, y)), robust_sigma(x)))
            line += f"{bias:+7.1f} {robust_sigma(z):6.1f} {np.median(np.hypot(x, y)):6.1f} {robust_sigma(x):5.1f} "
        print(line)
    print()
    for a in args.algos:
        s = np.array(summ[a])
        if len(s):
            inner = np.abs(s[:, 0]) <= 550
            print(f"algo {a}: |z|<=550 cm: mean |bias| {np.mean(np.abs(s[inner, 1])):.1f} cm, rms bias {np.sqrt(np.mean(s[inner, 1] ** 2)):.1f} cm, "
                  f"mean sigma_z {np.mean(s[inner, 2]):.1f} cm, mean sigma_x {np.mean(s[inner, 4]):.1f} cm, mean rho {np.mean(s[inner, 3]):.1f} cm")
    # energy versus position (last algorithm)
    a_e = args.algos[-1]
    esum = []
    print(f"\nenergy with the vertex of algo {a_e} (median over source events; 60Co = 2.506 MeV):")
    print(f"{'run':>5s} {'z':>6s} {'E_hit':>7s} {'E_chg':>7s} {'sig_hit%':>8s} {'sig_chg%':>8s}")
    for run in np.unique(tab["run"]):
        m = (tab["run"] == run) & (tab["algo"] == a_e) & tab["ok"] & np.isfinite(tab["e_hit"])
        if m.sum() < 5:
            continue
        eh = tab["e_hit"][m]; ec = tab["e_charge"][m]
        esum.append((tab["ztrue"][m][0], np.median(eh), np.median(ec), robust_sigma(eh) / np.median(eh), robust_sigma(ec) / np.median(ec)))
        print(f"{run:5d} {tab['ztrue'][m][0]:+6.0f} {np.median(eh):7.3f} {np.median(ec):7.3f} {100 * esum[-1][3]:8.1f} {100 * esum[-1][4]:8.1f}")
    esum = np.array(esum)
    if len(esum):
        inner = np.abs(esum[:, 0]) <= 550
        print(f"|z|<=550: E_hit spread (rms/mean) {100 * esum[inner, 1].std() / esum[inner, 1].mean():.1f}%, "
              f"E_charge spread {100 * esum[inner, 2].std() / esum[inner, 2].mean():.1f}%; "
              f"mean resolution E_hit {100 * esum[inner, 3].mean():.1f}%, E_charge {100 * esum[inner, 4].mean():.1f}%")
        np.save(args.output + "_energy.npy", esum)
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 4, figsize=(21, 5))
        colors = {"A": "C0", "B": "C1", "C": "C3", "D": "C2", "E": "C4"}
        labels = {"A": "window fitter, single v", "B": "window fitter, 2 media", "C": "ML fitter, time only",
                  "D": "ML fitter, time + hit pattern", "E": "ML fitter, time + charge"}
        for a in args.algos:
            s = np.array(summ[a])
            if not len(s):
                continue
            o = np.argsort(s[:, 0])
            axs[0].plot(s[o, 0], s[o, 1], "o-", color=colors[a], label=labels[a])
            axs[1].plot(s[o, 0], s[o, 2], "o-", color=colors[a], label=labels[a] + " (z)")
            axs[1].plot(s[o, 0], s[o, 4], "s--", color=colors[a], alpha=0.6, label=labels[a] + " (x)")
            axs[2].plot(s[o, 0], s[o, 3], "o-", color=colors[a], label=labels[a])
        axs[0].axhline(0, color="k", lw=0.5); axs[0].set_ylabel("z bias [cm]"); axs[0].set_ylim(-60, 60)
        axs[1].set_ylabel("robust sigma [cm]"); axs[1].set_ylim(0, 40)
        axs[2].set_ylabel("median rho [cm]"); axs[2].set_ylim(0, 60)
        if len(esum):
            o = np.argsort(esum[:, 0])
            axs[3].plot(esum[o, 0], esum[o, 1], "o-", color="C3", label=f"E_hit (algo {a_e}{', joint fit' if a_e in 'DE' else ''})")
            axs[3].plot(esum[o, 0], esum[o, 2], "s--", color="C0", label="E_charge")
            axs[3].axhline(2.506, color="k", lw=0.5); axs[3].set_ylim(2.0, 3.0); axs[3].set_ylabel("median energy [MeV]")
        for ax in axs:
            ax.set_xlabel("source z [cm]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("60Co z-scan (March 2003): vertex reconstruction vs. source position")
        fig.tight_layout()
        fig.savefig(args.output + ".png", dpi=110)
        print("plot:", args.output + ".png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
