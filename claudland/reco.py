"""Event reconstruction pipeline for one KamLAND run file.

Typical use::

    from claudland.reco import EventReconstructor
    rec = EventReconstructor("run_002279_000000_000001.sfz")
    rec.prepare()                       # pedestals, clock periods, run header
    table = rec.run(max_events=1000)    # numpy structured array, one row per physics event
    ev = rec[500]                       # RecoEvent (vertex, energy, hits) of physics event 500

Processing steps per event (``reconstruct``):

1. decompress the ``CmpATWD`` / ``CmpAntiATWD`` waveforms (:mod:`claudland.wfcomp`);
2. subtract the per-channel pedestal and the per-waveform baseline
   (:mod:`claudland.pedestal`);
3. find pulses and extract time and charge (:mod:`claudland.tq`, :mod:`claudland.calib`);
4. fit the vertex from the hit times (:mod:`claudland.vertex`);
5. estimate the visible energy from the charge and the hit pattern
   (:mod:`claudland.energy`).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from .sf import SFReader, SFEvent
from .banks import Header, RunHeader, HitHeader, decode_header, decode_run_header, decode_hit_header
from .wfcomp import WaveformDecompressor, Waveform, WaveformBatch, NSAMPLES
from .pedestal import PedestalManager, baseline_batch
from .calib import TQCalibration, ClockAnalyzer
from .tq import fast_tq
from .geometry import PMTTable, N_ID, N_ID17
from .vertex import VertexFitter, VertexResult
from .energy import EnergyEstimator, EnergyResult
from . import trigger as trg

__all__ = ["EventReconstructor", "RecoEvent", "HIT_DTYPE", "EVENT_DTYPE"]

HIT_DTYPE = np.dtype([
    ("cable", np.int16), ("atwd", np.int8), ("gain", np.int8), ("launch", np.int8),
    ("t", np.float32),        # hit time relative to the trigger (ns), T0 corrected
    ("q", np.float32),        # charge (p.e.), all pulses in the waveform
    ("q_first", np.float32),  # charge of the first pulse (p.e.)
    ("t_lead", np.float32),   # leading edge (sample)
    ("t_cfd", np.float32),    # constant-fraction time (sample)
    ("t_peak", np.float32),   # peak time (sample)
    ("height", np.float32),   # first-pulse height (ADC)
    ("npulse", np.int8), ("saturated", np.bool_), ("primary", np.bool_),
    ("tau", np.float32),      # ToF-corrected time minus event time (ns); NaN if no vertex
    ("used", np.bool_),       # used in the vertex fit window
])

EVENT_DTYPE = np.dtype([
    ("index", np.int32), ("event", np.int32), ("run", np.int32),
    ("unix_time", np.float64), ("timestamp", np.int64), ("trigger", np.int64),
    ("nsum", np.int16), ("nsum_max", np.int16),
    ("nhit", np.int16), ("nhit17", np.int16), ("nhit20", np.int16), ("nhit_od", np.int16),
    ("nwave", np.int16), ("nsat", np.int16),
    ("q_total", np.float32), ("q17", np.float32), ("q20", np.float32), ("q_od", np.float32),
    ("x", np.float32), ("y", np.float32), ("z", np.float32), ("t0", np.float32),
    ("r", np.float32), ("sigma_t", np.float32), ("n_used", np.int16), ("vertex_ok", np.bool_),
    ("e_charge", np.float32), ("e_hit", np.float32), ("q_window", np.float32),
    ("n_window", np.int16), ("dark_hits", np.float32), ("f_collect", np.float32),
])


@dataclass
class RecoEvent:
    """One reconstructed event: header, hit array (ID + OD), vertex and energy results."""
    header: Header
    hits: np.ndarray                 #: structured array (HIT_DTYPE), ID + OD
    vertex: Optional[VertexResult]
    energy: Optional[EnergyResult]
    index: int = -1                  #: sequential index of the event in the file

    @property
    def id_hits(self) -> np.ndarray:
        """The inner-detector part of the hit array."""
        return self.hits[self.hits["cable"] < N_ID]

    @property
    def od_hits(self) -> np.ndarray:
        """The outer-detector part of the hit array."""
        return self.hits[self.hits["cable"] >= N_ID]

    def record(self) -> np.ndarray:
        """One EVENT_DTYPE row."""
        h = self.header
        idh = self.id_hits
        prim = idh[idh["primary"]]
        od = self.od_hits
        rec = np.zeros(1, dtype=EVENT_DTYPE)[0]
        rec["index"] = self.index; rec["event"] = h.event_number; rec["run"] = h.run
        rec["unix_time"] = h.time_s; rec["timestamp"] = h.timestamp; rec["trigger"] = h.trigger_type & 0xFFFFFFFF
        rec["nsum"] = h.nsum; rec["nsum_max"] = h.nsum_max
        rec["nhit"] = len(prim); rec["nhit17"] = int((prim["cable"] < N_ID17).sum()); rec["nhit20"] = int((prim["cable"] >= N_ID17).sum())
        rec["nhit_od"] = int(od["primary"].sum()); rec["nwave"] = len(idh); rec["nsat"] = int(idh["saturated"].sum())
        rec["q_total"] = prim["q"].sum(); rec["q17"] = prim["q"][prim["cable"] < N_ID17].sum()
        rec["q20"] = prim["q"][prim["cable"] >= N_ID17].sum(); rec["q_od"] = od["q"][od["primary"]].sum()
        v = self.vertex
        if v is not None:
            rec["x"], rec["y"], rec["z"], rec["t0"] = v.x, v.y, v.z, v.t0
            rec["r"] = v.r; rec["sigma_t"] = v.sigma_t; rec["n_used"] = v.n_used; rec["vertex_ok"] = v.ok
        else:
            rec["x"] = rec["y"] = rec["z"] = rec["t0"] = rec["r"] = rec["sigma_t"] = np.nan
        e = self.energy
        if e is not None:
            rec["e_charge"], rec["e_hit"], rec["q_window"] = e.e_charge, e.e_hit, e.q_window
            rec["n_window"], rec["dark_hits"], rec["f_collect"] = e.n_window, e.dark_hits, e.f_collect
        else:
            rec["e_charge"] = rec["e_hit"] = rec["q_window"] = rec["f_collect"] = rec["dark_hits"] = np.nan
        return rec


class EventReconstructor:
    """Reconstruct the physics events of one ``.sfz`` (or ``.sf``) file."""

    def __init__(self, path: str, pmts: Optional[PMTTable] = None, calib: Optional[TQCalibration] = None,
                 vertex: Optional[VertexFitter] = None, energy: Optional[EnergyEstimator] = None,
                 gains: Sequence[int] = (0, 1, 2), time_key: str = "t_cfd", min_height: float = 5.0,
                 verbose: bool = True):
        self.path = os.fspath(path)
        self.reader = SFReader(self.path)
        self.pmts = pmts or PMTTable.load()
        self.calib = calib or TQCalibration()
        self.vertex_fitter = vertex or VertexFitter(self.pmts)
        self.energy_estimator = energy or EnergyEstimator(self.pmts)
        self.decoder = WaveformDecompressor()
        self.pedestals = PedestalManager()
        self.clock = ClockAnalyzer()
        self.gains = tuple(gains)
        self.time_key = time_key
        self.min_height = min_height
        self.verbose = verbose
        self.run_header: Optional[RunHeader] = None
        self.first_physics_index: Optional[int] = None
        self.n_pedestal_events = 0
        self.n_clock_events = 0
        self.occupancy = np.zeros(2144, dtype=np.int64)
        self.n_physics_seen = 0
        self._prepared = False
        self.timing: Dict[str, float] = {}

    # -- run-level preparation -------------------------------------------------
    def log(self, *a):
        """Print *msg* when verbose."""
        if self.verbose:
            print(*a, file=sys.stderr, flush=True)

    def prepare(self, max_scan: int = 2000, n_occupancy: int = 300) -> None:
        """Read the calibration block at the start of the run.

        Collects the compression constants (first event), the ``RunHeader``,
        pedestal-trigger and clock-trigger events; then scans the first
        ``n_occupancy`` physics events to flag dead channels.
        """
        t_start = time.time()
        rd = self.reader
        rd.rewind()
        first = rd.next()
        if first is None:
            raise EOFError("empty file")
        self.decoder.load_constants(first)
        if self.decoder.ready:
            self.pedestals.set_compression_pedestals(self.decoder.connection, self.decoder.pedestals)
        if "RunHeader" in first:
            self.run_header = decode_run_header(first["RunHeader"])
            self.log(f"run {self.run_header.run}: type={self.run_header.run_type!r} comment={self.run_header.comment!r} "
                     f"shifters={self.run_header.shifters!r}")
        n_phys = 0
        for i in range(max_scan):
            ev = rd.next()
            if ev is None:
                break
            h = decode_header(ev["Header"])
            if trg.is_pedestal(h.trigger_type):
                self.pedestals.add_event(self.decoder.decompress_arrays(ev, "ID", gains=self.gains))
                self.pedestals.add_event(self.decoder.decompress_arrays(ev, "OD", gains=self.gains))
                self.n_pedestal_events += 1
            elif trg.is_clock(h.trigger_type):
                self.clock.add_event(self.decoder.decompress_arrays(ev, "ID", gains=(3,)))
                self.clock.add_event(self.decoder.decompress_arrays(ev, "OD", gains=(3,)))
                self.n_clock_events += 1
            elif h.is_physics and "HitHeader" in ev:
                if self.first_physics_index is None:
                    self.first_physics_index = ev.index
                hh = decode_hit_header(ev["HitHeader"])
                np.add.at(self.occupancy, hh.cable, 1)
                n_phys += 1
                if n_phys >= n_occupancy:
                    break
        if self.clock.n_measured > 0:
            self.clock.apply(self.calib)
        self.n_physics_seen = n_phys
        if n_phys > 0:
            live = np.zeros(N_ID, dtype=bool)
            live[:] = self.occupancy[:N_ID] > 0
            self.energy_estimator.live = live
            self.calib.bad[:N_ID] = ~live
        self._prepared = True
        self.timing["prepare"] = time.time() - t_start
        self.log(f"prepare: {self.n_pedestal_events} pedestal events, {self.n_clock_events} clock events, "
                 f"{n_phys} physics events scanned in {self.timing['prepare']:.1f} s")
        self.log("  " + self.pedestals.summary())
        self.log("  " + self.calib.summary())
        if n_phys > 0:
            self.log(f"  live ID channels: {int(self.energy_estimator.live.sum())}/{N_ID}")

    # -- per-event processing ------------------------------------------------------
    def waveforms(self, event: SFEvent, detector: str = "ID") -> WaveformBatch:
        """Decoded waveforms of the event for one detector, as a :class:`~claudland.wfcomp.WaveformBatch`."""
        return self.decoder.decompress_arrays(event, detector, gains=self.gains)

    def hits_from_waveforms(self, wfs) -> np.ndarray:
        """Pedestal/baseline subtraction + TQ + calibration -> HIT_DTYPE array.

        *wfs* is a :class:`~claudland.wfcomp.WaveformBatch` or a list of
        :class:`~claudland.wfcomp.Waveform`.
        """
        b = WaveformBatch.from_list(wfs)
        n = len(b)
        out = np.zeros(n, dtype=HIT_DTYPE)
        if n == 0:
            return out
        raw = b.samples.astype(np.float64)
        cable = b.cable.astype(np.int64); atwd = b.atwd.astype(np.int64)
        gain = b.gain.astype(np.int64); launch = b.launch.astype(np.int64)
        ped = self.pedestals.get_batch(cable, atwd, gain)
        sat = raw >= 1023
        W = raw - ped
        W -= baseline_batch(W)[:, None]
        W[sat] = 0.0            # saturated samples carry no information; keep them off the charge
        r = fast_tq(W, sat, min_height=self.min_height)
        tsample = r[self.time_key]
        out["cable"] = cable; out["atwd"] = atwd; out["gain"] = gain; out["launch"] = launch
        out["t"] = self.calib.time_ns(cable, atwd, tsample, launch)
        out["q"] = self.calib.charge_pe(cable, atwd, gain, r["q_total"])
        out["q_first"] = self.calib.charge_pe(cable, atwd, gain, r["q_first"])
        out["t_lead"] = r["t_lead"]; out["t_cfd"] = r["t_cfd"]; out["t_peak"] = r["t_peak"]
        out["height"] = r["height"]; out["npulse"] = r["npulse"]; out["saturated"] = sat.any(axis=1)
        out["tau"] = np.nan
        # primary = earliest found hit per cable (prefer high gain, then earliest time)
        found = r["found"]
        order = np.lexsort((np.where(np.isfinite(out["t"]), out["t"], np.inf), gain, ~found, cable))
        c_sorted = cable[order]
        firstmask = np.concatenate(([True], c_sorted[1:] != c_sorted[:-1]))
        prim = np.zeros(n, dtype=bool)
        prim[order[firstmask]] = True
        out["primary"] = prim & found
        return out

    def reconstruct(self, event: SFEvent, fit_vertex: bool = True) -> RecoEvent:
        """Hits, vertex and energy of one event (a :class:`RecoEvent`)."""
        if not self._prepared:
            self.prepare()
        h = decode_header(event["Header"])
        hits_id = self.hits_from_waveforms(self.waveforms(event, "ID"))
        hits_od = self.hits_from_waveforms(self.waveforms(event, "OD"))
        hits = np.concatenate([hits_id, hits_od])
        vertex = energy = None
        if fit_vertex and len(hits_id):
            p = hits_id["primary"] & (hits_id["cable"] < N_ID)
            vertex = self.vertex_fitter.fit(hits_id["cable"][p], hits_id["t"][p], hits_id["q"][p])
            if np.isfinite(vertex.x):
                idm = hits["cable"] < N_ID
                tau = self.vertex_fitter.residuals(vertex, hits["cable"][idm], hits["t"][idm])
                hits["tau"][idm] = tau
                hits["used"][idm] = np.abs(tau) <= self.vertex_fitter.window_final
                pr = idm & hits["primary"]
                energy = self.energy_estimator.estimate(hits["cable"][pr], hits["tau"][pr], hits["q"][pr], vertex.xyz)
        return RecoEvent(h, hits, vertex, energy, event.index)

    # -- iteration -----------------------------------------------------------------
    def physics_events(self, start: Optional[int] = None, max_events: Optional[int] = None) -> Iterator[SFEvent]:
        """Iterate over events that carry waveforms and a physics trigger."""
        if not self._prepared:
            self.prepare()
        rd = self.reader
        offsets = rd.build_index()
        i0 = self.first_physics_index if start is None else start
        n = 0
        for i in range(i0 or 0, len(offsets)):
            ev = rd.read_at(offsets[i], i)
            h = decode_header(ev["Header"])
            if not (h.is_physics and "HitHeader" in ev):
                continue
            yield ev
            n += 1
            if max_events is not None and n >= max_events:
                return

    def run(self, max_events: Optional[int] = None, start: Optional[int] = None,
            keep_hits: bool = False, progress_every: int = 500) -> np.ndarray:
        """Reconstruct physics events; returns an EVENT_DTYPE structured array.

        With ``keep_hits=True`` the per-event hit arrays are stored in
        ``self.hit_arrays`` (list, same order as the table)."""
        t_start = time.time()
        rows: List[np.ndarray] = []
        self.hit_arrays: List[np.ndarray] = []
        for n, ev in enumerate(self.physics_events(start, max_events), 1):
            rec = self.reconstruct(ev)
            rows.append(rec.record())
            if keep_hits:
                self.hit_arrays.append(rec.hits)
            if progress_every and n % progress_every == 0:
                self.log(f"  {n} events, {time.time() - t_start:.1f} s")
        table = np.array(rows, dtype=EVENT_DTYPE) if rows else np.zeros(0, dtype=EVENT_DTYPE)
        self.timing["run"] = time.time() - t_start
        self.log(f"run: {len(table)} physics events reconstructed in {self.timing['run']:.1f} s")
        return table

    def __getitem__(self, i: int) -> RecoEvent:
        """Reconstruct the event with sequential file index ``i``."""
        return self.reconstruct(self.reader[i])

    # -- calibration from data --------------------------------------------------------
    def calibrate_t0(self, hit_arrays: Sequence[np.ndarray], min_hits: int = 20,
                     fixed_vertex: Optional[Sequence[float]] = None, window: float = 15.0) -> np.ndarray:
        """Per-(cable, ATWD) time offsets from ToF-corrected residuals.

        With ``fixed_vertex=None`` the residuals ``tau`` stored by the vertex
        fit (hits inside the fit window) are used -- this removes channel-to-
        channel offsets but cannot correct a global position bias.  With a
        ``fixed_vertex`` (cm), e.g. the known position of a calibration
        source, the time of flight is recomputed from that point and the event
        time is taken as the peak of the corrected-time distribution, so the
        resulting offsets also absorb light-propagation biases.  The median
        residual per channel is used; the mean over channels is removed so the
        event-time reference does not move.  Adds to and returns
        ``self.calib.t0``."""
        cabs, atws, taus = [], [], []
        for h in hit_arrays:
            idm = (h["cable"] < N_ID) & h["primary"] & np.isfinite(h["t"])
            if fixed_vertex is None:
                m = idm & h["used"]
                cabs.append(h["cable"][m]); atws.append(h["atwd"][m]); taus.append(h["tau"][m])
            else:
                r = np.asarray(fixed_vertex, dtype=float)
                tof, u, d = self.vertex_fitter.tof(r, self.pmts.xyz[h["cable"][idm]])
                tau = h["t"][idm] - tof
                if len(tau) < 4:
                    continue
                T = VertexFitter._peak(tau)
                T = np.mean(tau[np.abs(tau - T) <= window])
                m = np.abs(tau - T) <= window
                cabs.append(h["cable"][idm][m]); atws.append(h["atwd"][idm][m]); taus.append(tau[m] - T)
        cab = np.concatenate(cabs); atw = np.concatenate(atws); tau = np.concatenate(taus)
        t0 = np.zeros_like(self.calib.t0)
        key = cab.astype(np.int64) * 2 + atw
        order = np.argsort(key, kind="stable")
        key_s = key[order]; tau_s = tau[order]
        bounds = np.flatnonzero(np.concatenate(([True], key_s[1:] != key_s[:-1], [True])))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= min_hits:
                k = key_s[a]
                t0[k // 2, k % 2] = np.median(tau_s[a:b])
        nz = t0 != 0
        if nz.any():
            t0[nz] -= np.mean(t0[nz])
        self.calib.t0 += t0
        self.calib.meta["t0_source"] = "fixed_vertex" if fixed_vertex is not None else "fitted_vertex"
        return t0

    def calibrate_q1pe(self, hit_arrays: Sequence[np.ndarray], min_hits: int = 100) -> np.ndarray:
        """Per-(cable, ATWD) single-p.e. charge from the mode of the hit charge spectrum
        (high-gain waveforms only).  Updates ``self.calib.q1pe`` and returns it."""
        cab = np.concatenate([h["cable"] for h in hit_arrays]); atw = np.concatenate([h["atwd"] for h in hit_arrays])
        gain = np.concatenate([h["gain"] for h in hit_arrays]); q = np.concatenate([h["q"] for h in hit_arrays])
        # back to ADC using the current calibration
        adc = q * self.calib.q1pe[cab, atw] / self.calib.gain_factor[gain]
        sel = (gain == 0) & (adc > 30) & (adc < 800)
        key = cab[sel].astype(np.int64) * 2 + atw[sel]; adc = adc[sel]
        order = np.argsort(key, kind="stable"); key = key[order]; adc = adc[order]
        bounds = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1], [True])))
        edges = np.arange(30, 800, 10.0)
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= min_hits:
                hcount, _ = np.histogram(adc[a:b], bins=edges)
                hs = np.convolve(hcount, np.ones(5) / 5, mode="same")
                k = int(hs.argmax())
                # parabolic refinement of the mode
                if 0 < k < len(hs) - 1:
                    y0, y1, y2 = hs[k - 1], hs[k], hs[k + 1]
                    den = y0 - 2 * y1 + y2
                    dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
                else:
                    dk = 0.0
                kk = key[a]
                self.calib.q1pe[kk // 2, kk % 2] = edges[k] + 5.0 + dk * 10.0
        return self.calib.q1pe
