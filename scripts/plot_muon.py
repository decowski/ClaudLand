#!/usr/bin/env python3
"""Muon event displays and sample plots from the output of ``muon_fit.py``.

    python3 scripts/plot_muon.py cache/muons_1467.npz --event 566 -o ev566_mu.png   # one event display
    python3 scripts/plot_muon.py cache/muons_1467.npz -o muons_1467.png              # sample summary
    python3 scripts/plot_muon.py cache/muons_*.npz -o muons_all.png                  # several files

Event display: unrolled PMT sphere (azimuth vs cos(polar angle)) with the hit
tubes coloured by hit time (left) and charge (right); the fitted track is drawn
as its radial projection on the sphere (entrance ▲, exit ▼); bottom: residual
histogram and measured vs predicted first-light time.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from claudland.geometry import PMTTable, N_ID, N_ID17
from claudland.muon import MuonTrack, MuonTrackFitter, R_PMT, R_LS, first_light_time, unit

DQDX_LS_REF, DQDX_BO_REF = 629.4, 31.45   # p.e./cm (Ichimura, Abe, Keefer)


def load(files):
    """Concatenate the muon tables (and hit arrays) of several ``muon_fit.py`` outputs."""
    tabs, hits, hev, res = [], [], [], []
    off = 0
    for f in files:
        d = np.load(f)
        tabs.append(d["muons"])
        if "hits" in d:
            hits.append(d["hits"]); hev.append(d["hit_event"] + off); res.append(d["residual"])
        off += len(d["muons"])
    # tables written by different versions may differ in their trailing columns: keep the common ones
    common = [n for n in tabs[0].dtype.names if all(n in t.dtype.names for t in tabs)]
    if any(len(t.dtype.names) != len(common) for t in tabs):
        import numpy.lib.recfunctions as rf
        tabs = [rf.repack_fields(t[common]) for t in tabs]
    tab = np.concatenate(tabs)
    if hits:
        return tab, np.concatenate(hits), np.concatenate(hev), np.concatenate(res)
    return tab, None, None, None


def sphere_coords(xyz):
    """(azimuth in degrees, cos polar angle) of points on the PMT sphere."""
    r = np.linalg.norm(xyz, axis=1)
    return np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), xyz[:, 2] / r


def track_projection(row, n=200):
    """Radial projection of the track onto the PMT sphere, for drawing."""
    e = np.array([row["ex"], row["ey"], row["ez"]]); x = np.array([row["xx"], row["xy"], row["xz"]])
    s = np.linspace(0, 1, n)
    pts = e[None, :] + s[:, None] * (x - e)[None, :]
    pts = pts / np.linalg.norm(pts, axis=1)[:, None] * R_PMT
    return pts


def event_display(tab, hits, hev, res, k, pmts, out):
    """Draw one muon event (hit map by time and charge, residuals, measured vs predicted times)."""
    row = tab[k]
    h = hits[hev == k]; r = res[hev == k]
    idm = h["cable"] < N_ID
    hid = h[idm]; rid = r[idm]
    xyz = pmts.xyz[hid["cable"]]
    phi, cz = sphere_coords(xyz)
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    t_rel = hid["t"] - np.nanpercentile(hid["t"], 2)
    used = np.abs(rid) < 8
    for ax, val, label, cmap, vmin, vmax in ((axs[0, 0], t_rel, "hit time − earliest (ns)", "jet", 0, 120),
                                             (axs[0, 1], np.log10(np.clip(hid["q"], 0.1, None)), "log10 charge (p.e.)", "viridis", 0, 4)):
        sc = ax.scatter(phi, cz, c=val, s=8 + 3 * np.log10(np.clip(hid["q"], 1, None)) ** 2, cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors=np.where(used, "none", "k"), linewidths=0.4)
        plt.colorbar(sc, ax=ax, label=label)
        if row["ok"]:
            pr = track_projection(row)
            pphi, pcz = sphere_coords(pr)
            brk = np.flatnonzero(np.abs(np.diff(pphi)) > 180) + 1
            for seg_phi, seg_cz in zip(np.split(pphi, brk), np.split(pcz, brk)):
                ax.plot(seg_phi, seg_cz, "k--", lw=1)
            ax.plot(pphi[0], pcz[0], "k^", ms=12, mfc="w", label="entrance"); ax.plot(pphi[-1], pcz[-1], "kv", ms=12, mfc="w", label="exit")
            ax.legend(loc="lower left", fontsize=8)
        od = h[~idm]
        if len(od):
            ophi, ocz = sphere_coords(pmts.xyz[od["cable"]])
            ax.scatter(ophi, ocz, marker="s", s=25, facecolors="none", edgecolors="m", lw=0.8, label="OD hits")
        ax.set_xlim(-180, 180); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("cos(polar angle)")
    axs[0, 0].set_title(f"run {row['run']} event {row['event']} (index {row['index']}): nhit={row['nhit']} nOD={row['nhit_od']} "
                        f"Q17={row['q17']:.3g} Q20={row['q20']:.3g} p.e.", fontsize=10)
    axs[0, 1].set_title(f"cosZ={row['cos_zenith']:.2f} b={row['impact']:.0f} L_LS={row['l_ls']:.0f} L_BO={row['l_bo']:.0f} cm "
                        f"n={row['n_eff']:.3f} sigma_t={row['sigma_t']:.2f} ns used {row['n_used']}/{row['nhit']} "
                        f"{'converged' if row['converged'] else 'NOT converged'}", fontsize=10)
    ax = axs[1, 0]
    fin = np.isfinite(rid)
    ax.hist(np.clip(rid[fin], -30, 100), bins=130, range=(-30, 100), histtype="step", color="k", label="all ID tubes")
    big = fin & (hid["q"] > 100)
    ax.hist(np.clip(rid[big], -30, 100), bins=130, range=(-30, 100), histtype="stepfilled", alpha=0.4, label="q > 100 p.e.")
    ax.axvline(0, color="r", lw=0.8); ax.set_xlabel("t − t_first-light (ns)"); ax.set_ylabel("tubes"); ax.legend()
    ax = axs[1, 1]
    pred = hid["t"] - rid
    ax.scatter(pred - row["t0"], hid["t"] - row["t0"], c=np.log10(np.clip(hid["q"], 0.1, None)), s=6, cmap="viridis", vmin=0, vmax=4)
    lo, hi = np.nanpercentile(pred - row["t0"], [0, 100])
    ax.plot([lo, hi], [lo, hi], "r-", lw=0.8)
    ax.set_xlabel("predicted first-light time − t0 (ns)"); ax.set_ylabel("measured hit time − t0 (ns)")
    ax.set_ylim(lo - 20, hi + 120)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print("written", out)


def summary(tab, out, title=""):
    """Draw the sample plots (charge vs track length, zenith, impact parameter, sigma_t)."""
    ok = tab["ok"] & tab["converged"]
    t = tab[ok]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ax = axs[0, 0]
    ls = t["l_ls"] > 0
    ax.scatter(t["l_ls"][ls], t["q17"][ls], s=12, label="LS-crossing")
    L = np.linspace(0, 1300, 10)
    ax.plot(L, DQDX_LS_REF * L + DQDX_BO_REF * 360, "r--", label=f"{DQDX_LS_REF:.0f} p.e./cm × L_LS + Cherenkov")
    if ls.sum() > 3:
        a = np.median((t["q17"][ls] - DQDX_BO_REF * t["l_bo"][ls]) / t["l_ls"][ls])
        ax.plot(L, a * L + DQDX_BO_REF * 360, "k-", lw=0.8, label=f"median (Q17−31.45·L_BO)/L_LS = {a:.0f} p.e./cm")
    ax.set_xlabel("track length in LS (cm)"); ax.set_ylabel("Q17 (p.e.)"); ax.legend(fontsize=8); ax.set_ylim(0, 1.5e6)
    ax = axs[0, 1]
    bo = ~ls & (t["l_bo"] > 50)
    ax.scatter(t["l_bo"][bo], t["q17"][bo], s=12, color="C1", label="buffer-oil only")
    L = np.linspace(0, 1100, 10); ax.plot(L, DQDX_BO_REF * L, "r--", label=f"{DQDX_BO_REF:.1f} p.e./cm")
    if bo.sum() > 3:
        a = np.median(t["q17"][bo] / t["l_bo"][bo]); ax.plot(L, a * L, "k-", lw=0.8, label=f"median Q17/L_BO = {a:.0f} p.e./cm")
    ax.set_xlabel("track length in buffer oil (cm)"); ax.set_ylabel("Q17 (p.e.)"); ax.legend(fontsize=8); ax.set_ylim(0, 1e5)
    ax = axs[0, 2]
    ax.hist(np.log10(np.clip(tab["q17"], 1, None)), bins=40, range=(2, 7), histtype="step", color="k", label="all selected")
    ax.hist(np.log10(np.clip(t["q17"][ls], 1, None)), bins=40, range=(2, 7), histtype="stepfilled", alpha=0.4, label="fit: LS")
    ax.hist(np.log10(np.clip(t["q17"][~ls], 1, None)), bins=40, range=(2, 7), histtype="stepfilled", alpha=0.4, label="fit: oil only")
    ax.set_xlabel("log10 Q17 (p.e.)"); ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.hist(t["cos_zenith"], bins=20, range=(-1, 1), histtype="step", color="k", label="all")
    ax.hist(t["cos_zenith"][ls], bins=20, range=(-1, 1), histtype="stepfilled", alpha=0.4, label="LS")
    ax.set_xlabel("cos(zenith) of the track (1 = down-going)"); ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.hist((t["impact"] / R_PMT) ** 2, bins=25, range=(0, 1), histtype="step", color="k")
    ax.axvline((R_LS / R_PMT) ** 2, color="r", ls="--", label="balloon")
    ax.set_xlabel("(impact parameter / 850 cm)²  (flat for an isotropic flux)"); ax.legend(fontsize=8)
    ax = axs[1, 2]
    ax.scatter(t["impact"], t["sigma_t"], s=10, c=np.log10(np.clip(t["q17"], 1, None)), cmap="viridis")
    ax.set_xlabel("impact parameter (cm)"); ax.set_ylabel("sigma_t of used hits (ns)"); ax.set_ylim(0, 8)
    fig.suptitle(f"{title} {len(tab)} muon candidates, {ok.sum()} converged tracks, {ls.sum()} through the LS", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print("written", out)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--event", type=int, help="file event index (column 'index') to display")
    ap.add_argument("-o", "--output", default="muons.png")
    ap.add_argument("--od-trigger", action="store_true", help="drop Q17 < 10000 p.e. events without an OD trigger bit")
    ap.add_argument("--min-q17", type=float, default=0.0)
    args = ap.parse_args()
    tab, hits, hev, res = load(args.files)
    keep = tab["q17"] >= args.min_q17
    if args.od_trigger:
        keep &= (tab["q17"] >= 10000) | ((tab["trigger"] & 0x00F00000) != 0)
    if not keep.all():
        idx = np.flatnonzero(keep)
        remap = -np.ones(len(tab), dtype=int); remap[idx] = np.arange(len(idx))
        tab = tab[keep]
        if hits is not None:
            hk = keep[hev]; hits, hev, res = hits[hk], remap[hev[hk]], res[hk]
    pmts = PMTTable.load()
    if args.event is not None:
        k = int(np.flatnonzero(tab["index"] == args.event)[0])
        event_display(tab, hits, hev, res, k, pmts, args.output)
    else:
        summary(tab, args.output, title=os.path.basename(args.files[0]) if len(args.files) == 1 else "")


if __name__ == "__main__":
    main()
