#!/usr/bin/env python3
"""Event display: waveforms and hit pattern of one event, saved as PNG.

Examples::

    python scripts/plot_event.py run_002279_000000_000001.sfz --event 338 -o ev338.png
    python scripts/plot_event.py FILE --index 400 --cables 3 4 8 --raw
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from claudland.reco import EventReconstructor
from claudland.banks import decode_header
from claudland.geometry import N_ID, N_ID17


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--event", type=int, help="event number (as in the Header bank)")
    g.add_argument("--index", type=int, help="sequential index in the file")
    ap.add_argument("--cables", type=int, nargs="*", help="cables whose waveforms to draw (default: 8 highest-charge)")
    ap.add_argument("--raw", action="store_true", help="draw raw ADC waveforms instead of pedestal-subtracted")
    ap.add_argument("--calib", help="JSON calibration to apply")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from claudland.calib import TQCalibration
    rec = EventReconstructor(args.file, calib=TQCalibration.load(args.calib) if args.calib else None,
                             gains=(0,), verbose=False)
    rec.prepare()
    if args.index is not None:
        ev = rec.reader[args.index]
    else:
        ev = None
        for e in rec.reader:
            if e.event_number == args.event:
                ev = e
                break
        if ev is None:
            sys.exit(f"event {args.event} not found")
    h = decode_header(ev["Header"])
    r = rec.reconstruct(ev)
    wfs = rec.waveforms(ev, "ID")
    hits = r.id_hits
    prim = hits[hits["primary"]]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    # -- waveforms -------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    if args.cables:
        sel = [w for w in wfs if w.cable in args.cables]
    else:
        order = np.argsort(-hits["q"])[:8]
        want = set(hits["cable"][order].tolist())
        sel = [w for w in wfs if w.cable in want]
    for w in sel[:12]:
        if args.raw:
            y = w.samples.astype(float)
        else:
            y = w.samples.astype(float) - rec.pedestals.get(w.cable, w.atwd, w.gain)
            y -= np.median(np.sort(y)[:40])
        i = np.flatnonzero((hits["cable"] == w.cable) & (hits["atwd"] == w.atwd))
        lab = f"cable {w.cable} {'AB'[w.atwd]} launch {w.launch_offset}"
        if len(i):
            lab += f"  q={hits['q'][i[0]]:.2f} pe t={hits['t'][i[0]]:.0f} ns"
        ax.plot(y, drawstyle="steps-mid", lw=1, label=lab)
    ax.set_xlabel("sample (1 sample = %.2f ns)" % rec.calib.bin_ns[:N_ID].mean())
    ax.set_ylabel("ADC" if args.raw else "ADC - pedestal")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"run {h.run} event {h.event_number}  trigger {h.trigger_name}  nhit={len(prim)}")
    # -- time distribution -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    t = prim["t"][np.isfinite(prim["t"])]
    ax.hist(t, bins=80, histtype="step", label="hit time")
    if r.vertex is not None and np.isfinite(r.vertex.x):
        tau = prim["tau"][np.isfinite(prim["tau"])] + r.vertex.t0
        ax.hist(tau, bins=80, histtype="step", label="ToF corrected")
        ax.axvline(r.vertex.t0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time relative to trigger [ns]")
    ax.legend(fontsize=8)
    # -- hit map (theta, phi) ---------------------------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    P = rec.pmts.xyz[:N_ID]
    theta = np.degrees(np.arccos(P[:, 2] / np.linalg.norm(P, axis=1)))
    phi = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    ax.scatter(phi, theta, s=4, c="lightgrey")
    c = prim["cable"]
    sc = ax.scatter(phi[c], theta[c], s=12 + 20 * np.clip(prim["q"], 0, 5),
                    c=prim["tau"] if r.vertex is not None else prim["t"], cmap="viridis", vmin=-20, vmax=60)
    plt.colorbar(sc, ax=ax, label="ToF-corrected time - T0 [ns]")
    ax.set_xlabel("phi [deg]"); ax.set_ylabel("theta [deg] (0 = top)")
    ax.invert_yaxis()
    # -- summary text ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    lines = [f"run {h.run} event {h.event_number} (file index {ev.index})",
             f"unix {h.unix_time}.{h.microsec:06d}   timestamp {h.timestamp_s:.6f} s",
             f"trigger 0x{h.trigger_type & 0xFFFFFFFF:x}  nsum {h.nsum}  nsum_max {h.nsum_max}",
             f"ID hits {len(prim)} (17\" {int((prim['cable'] < N_ID17).sum())}, 20\" {int((prim['cable'] >= N_ID17).sum())})   OD hits {int(r.od_hits['primary'].sum())}",
             f"Q total {prim['q'].sum():.0f} p.e."]
    if r.vertex is not None:
        v = r.vertex
        lines += [f"vertex ({v.x:.0f}, {v.y:.0f}, {v.z:.0f}) cm  r={v.r:.0f}  ok={v.ok}",
                  f"T0 {v.t0:.1f} ns  sigma_t {v.sigma_t:.1f} ns  n_used {v.n_used}/{v.n_hits}  it {v.iterations}"]
    if r.energy is not None:
        e = r.energy
        lines += [f"E_charge {e.e_charge:.2f} MeV   E_hit {e.e_hit:.2f} MeV",
                  f"Q window {e.q_window:.0f} p.e.  n_window {e.n_window}  dark {e.dark_hits:.1f}  f_collect {e.f_collect:.3f}"]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    out = args.output or f"event_{h.run}_{h.event_number}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
