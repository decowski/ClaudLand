#!/usr/bin/env python3
"""Decode a run once and cache the per-hit TQ data (no vertex/energy fit).

The cache holds raw quantities (samples, ADC sums) plus the run's own ATWD
sampling periods, so any calibration (T0, Q0) and any vertex algorithm can be
applied afterwards without touching the 268 MB raw file again.

    python scripts/extract_hits.py run_002283_000000_000001.sfz -n 2500 -o cache/hits_2283.npz
    python scripts/extract_hits.py run_001518_00000?_00000[1-4].sf -o cache/hits_1518.npz   # several files of one run
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from claudland.reco import EventReconstructor
from claudland.banks import decode_header, decode_hit_header

CACHE_HIT_DTYPE = np.dtype([
    ("ev", np.int32), ("cable", np.int16), ("atwd", np.int8), ("launch", np.int8),
    ("t_cfd", np.float32), ("t_lead", np.float32), ("t_peak", np.float32),
    ("q_adc", np.float32), ("q_first_adc", np.float32), ("height", np.float32),
    ("npulse", np.int8), ("saturated", np.bool_), ("primary", np.bool_),
])
CACHE_EVENT_DTYPE = np.dtype([
    ("index", np.int32), ("event", np.int32), ("unix_time", np.float64), ("timestamp", np.int64),
    ("trigger", np.int64), ("nsum", np.int16), ("nsum_max", np.int16), ("nhit", np.int16), ("nhit_od", np.int16),
])


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="the run's files in order (only the first one has the calibration block)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-n", "--max-events", type=int, default=None)
    ap.add_argument("--min-nhit", type=int, default=0, help="skip physics events with fewer ID hits")
    args = ap.parse_args()

    t0 = time.time()
    rec = EventReconstructor(args.files, gains=(0,), verbose=True)
    rec.prepare()
    events, hits = [], []
    n = 0
    for ev in rec.physics_events():
        h = decode_header(ev["Header"])
        hh = decode_hit_header(ev["HitHeader"])
        if hh.nhit < args.min_nhit:
            continue
        hid = rec.hits_from_waveforms(rec.waveforms(ev, "ID"))
        hod = rec.hits_from_waveforms(rec.waveforms(ev, "OD"))
        for src in (hid, hod):
            c = np.zeros(len(src), dtype=CACHE_HIT_DTYPE)
            c["ev"] = n
            for k in ("cable", "atwd", "launch", "t_cfd", "t_lead", "t_peak", "height", "npulse", "saturated", "primary"):
                c[k] = src[k]
            # store ADC sums (undo the default 200-count normalisation of hits_from_waveforms)
            c["q_adc"] = src["q"] * rec.calib.q1pe[src["cable"], src["atwd"]] / rec.calib.gain_factor[src["gain"]]
            c["q_first_adc"] = src["q_first"] * rec.calib.q1pe[src["cable"], src["atwd"]] / rec.calib.gain_factor[src["gain"]]
            hits.append(c)
        events.append((ev.index, h.event_number, h.time_s, h.timestamp, h.trigger_type & 0xFFFFFFFF, h.nsum, h.nsum_max,
                       int(hid["primary"].sum()), int(hod["primary"].sum())))
        n += 1
        if args.max_events and n >= args.max_events:
            break
        if n % 500 == 0:
            print(f"  {n} events {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    events = np.array(events, dtype=CACHE_EVENT_DTYPE)
    hits = np.concatenate(hits) if hits else np.zeros(0, CACHE_HIT_DTYPE)
    rh = rec.run_header
    np.savez_compressed(args.output, events=events, hits=hits, bin_ns=rec.calib.bin_ns,
                        live=rec.energy_estimator.live, run=rh.run if rh else -1,
                        source_z_cm=rh.source_z_cm if rh else np.nan, run_comment=rh.comment if rh else "",
                        run_type=rh.run_type if rh else "",
                        pedestal_counts=rec.pedestals._n[:, :, 0])
    print(f"{args.output}: {len(events)} events, {len(hits)} hits, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
