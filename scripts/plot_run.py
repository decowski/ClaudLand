#!/usr/bin/env python3
"""Summary plots of a reconstructed run table (.npz from reco_run.py).

    python scripts/plot_run.py reco_2279.npz -o reco_2279.png
"""
import argparse
import sys

import numpy as np


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("-o", "--output")
    ap.add_argument("--source-z", type=float, help="true source z (cm) to mark")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.npz, allow_pickle=True)
    tab = d["events"]
    run = int(d["run"]) if "run" in d else -1
    ok = tab["vertex_ok"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    ax = axs[0, 0]; ax.hist(tab["nhit"], bins=100); ax.set_xlabel("ID hits"); ax.set_yscale("log")
    ax = axs[0, 1]
    q = tab["q_total"][ok]
    ax.hist(q, bins=100, range=(0, np.percentile(q, 99.5) * 1.2)); ax.set_xlabel("total charge [p.e.] (99.5% range)"); ax.set_yscale("log")
    ax = axs[0, 2]
    ax.hist(tab["e_charge"][ok], bins=100, range=(0, 6), histtype="step", label="E_charge")
    ax.hist(tab["e_hit"][ok], bins=100, range=(0, 6), histtype="step", label="E_hit")
    ax.set_xlabel("visible energy [MeV]"); ax.legend()
    ax = axs[1, 0]
    ax.hist(tab["z"][ok], bins=100, range=(-800, 800)); ax.set_xlabel("z [cm]")
    if args.source_z is not None:
        ax.axvline(args.source_z, color="r", ls="--")
    ax = axs[1, 1]
    rho2 = (tab["x"][ok] ** 2 + tab["y"][ok] ** 2) / 650.0 ** 2
    ax.hist2d(rho2, tab["z"][ok], bins=[60, 80], range=[[0, 1.6], [-800, 800]], cmin=1)
    ax.set_xlabel(r"$(\rho / r_{balloon})^2$"); ax.set_ylabel("z [cm]")
    ax = axs[1, 2]
    ax.hist2d(tab["x"][ok], tab["y"][ok], bins=80, range=[[-800, 800], [-800, 800]], cmin=1)
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]"); ax.set_aspect("equal")
    fig.suptitle(f"run {run}: {len(tab)} physics events, {ok.sum()} fitted   {d['run_type'] if 'run_type' in d else ''} {d['run_comment'] if 'run_comment' in d else ''}")
    fig.tight_layout()
    out = args.output or args.npz.replace(".npz", ".png")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
