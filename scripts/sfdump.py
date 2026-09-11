#!/usr/bin/env python3
"""Dump the structure of a KamLAND SF/SFZ file (Python version of SF/examples/sfdump.cc).

Examples::

    python scripts/sfdump.py run_002279_000000_000001.sfz            # first 5 events
    python scripts/sfdump.py FILE -n 20 --header                       # decoded Header banks
    python scripts/sfdump.py FILE --summary                            # trigger-type census of the whole file
    python scripts/sfdump.py FILE --event 338 --waveforms              # decoded waveforms of one event
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from claudland.sf import SFReader
from claudland.banks import decode_header, decode_run_header, decode_hit_header, decode_history
from claudland.wfcomp import WaveformDecompressor
from claudland import trigger as trg


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-n", "--events", type=int, default=5, help="number of events to dump (default 5)")
    ap.add_argument("--start", type=int, default=0, help="first event index")
    ap.add_argument("--event", type=int, help="dump only the event with this event number")
    ap.add_argument("--header", action="store_true", help="decode the Header bank")
    ap.add_argument("--history", action="store_true", help="decode the History bank")
    ap.add_argument("--hits", action="store_true", help="decode the HitHeader banks")
    ap.add_argument("--waveforms", action="store_true", help="decompress and print the waveforms")
    ap.add_argument("--summary", action="store_true", help="census of trigger types over the whole file")
    args = ap.parse_args()

    rd = SFReader(args.file)
    print(f"File: {args.file}  ({rd.file_size / 1e6:.1f} MB)")
    dec = WaveformDecompressor()

    if args.summary:
        cnt = collections.Counter()
        nhit = collections.defaultdict(list)
        first = last = None
        for ev in rd:
            h = decode_header(ev["Header"])
            key = h.trigger_type & 0xFFFFFFFF
            cnt[key] += 1
            nhit[key].append(decode_hit_header(ev["HitHeader"]).nhit if "HitHeader" in ev else 0)
            first = first or h
            last = h
        print(f"{len(rd)} events, run {first.run}, {last.unix_time - first.unix_time} s wall time")
        print(f"{'trigger':>10s}  {'name':45s} {'count':>7s} {'<nhit>':>8s} {'max':>5s}")
        for key, c in cnt.most_common():
            a = np.array(nhit[key])
            print(f"0x{key:08x}  {trg.describe(key):45s} {c:7d} {a.mean():8.1f} {a.max():5d}")
        return

    first = rd.next()
    if first is not None:
        dec.load_constants(first)
        if "RunHeader" in first:
            print("RunHeader:", decode_run_header(first["RunHeader"]))
    rd.rewind()
    shown = 0
    for i, ev in enumerate(rd):
        if args.event is not None:
            if ev.event_number != args.event:
                continue
        elif i < args.start:
            continue
        print(f"--- event #{i} run={ev.run} subrun={ev.subrun} event={ev.event_number} offset={ev.offset} size={ev.size}")
        for name, b in ev.banks.items():
            print(f"    {name:20s} size={b.size:9d} form={b.form[:50]!r:52s} endian={'big' if b.endian else 'little'}")
        if args.header or args.waveforms:
            h = decode_header(ev["Header"])
            print(f"    Header: trigger=0x{h.trigger_type & 0xFFFFFFFF:x} ({h.trigger_name}) nsum={h.nsum} nsum_max={h.nsum_max} "
                  f"unix={h.unix_time}.{h.microsec:06d} timestamp={h.timestamp} ({h.timestamp_s:.6f} s) "
                  f"dt={h.time_difference * 25e-9 * 1e6:.1f} us")
        if args.history and "History" in ev:
            print("    History:", decode_history(ev["History"]))
        if args.hits:
            for hn in ("HitHeader", "AntiHitHeader"):
                if hn in ev:
                    hh = decode_hit_header(ev[hn])
                    print(f"    {hn}: nhit={hh.nhit} waveforms={hh.total_waveforms} cables={hh.cable[:10].tolist()}...")
        if args.waveforms:
            for det in ("ID", "OD"):
                wfs = dec.decompress(ev, det)
                print(f"    {det}: {len(wfs)} waveforms")
                for w in wfs[:5]:
                    print(f"      cable {w.cable:4d} ATWD {'AB'[w.atwd]} gain {w.gain_name} launch {w.launch_offset:3d}: "
                          f"{w.samples[:16].tolist()} ... min {w.samples.min()} max {w.samples.max()}")
        shown += 1
        if args.event is not None or shown >= args.events:
            break


if __name__ == "__main__":
    main()
