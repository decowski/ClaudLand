"""Typed decoders for the standard KamLAND SF banks.

Field layouts follow ``AKat/RootSF/SFbank`` (``SFHeaderBank``,
``SFHitHeaderBank``, ``SFHistoryBank``) and the ``RunHeader`` written by the
event builder.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .sf import SFBank, SFEvent, expand_form
from . import trigger as trg

__all__ = ["Header", "RunHeader", "History", "HitHeader", "decode_header",
           "decode_run_header", "decode_history", "decode_hit_header"]


@dataclass
class Header:
    """The per-event ``Header`` bank (format ``2CIB4ILI8BIL[B]``)."""

    major_version: int
    minor_version: int
    run: int
    subrun: int
    event_number: int
    unix_time: int
    microsec: int
    nanosec: int
    timestamp: int          #: 40 MHz clock count (25 ns ticks)
    trigger_type: int
    nsum: int               #: inner-detector Nsum (number of 17" PMT hits at trigger)
    nsum_max: int
    nsum_veto_top: int
    nsum_veto_upper: int
    nsum_veto_lower: int
    nsum_veto_bottom: int
    nsum_5inch: int
    nsum_calibration: int
    error_status: int
    time_difference: int    #: 40 MHz ticks since previous trigger
    n_history: int = 0

    @property
    def time_s(self) -> float:
        """Unix time including the sub-second part, in seconds."""
        return self.unix_time + 1e-6 * self.microsec + 1e-9 * self.nanosec

    @property
    def timestamp_s(self) -> float:
        """The 40 MHz time stamp converted to seconds."""
        return self.timestamp * 25e-9

    @property
    def trigger_name(self) -> str:
        """Human-readable description of the trigger word (see :mod:`claudland.trigger`)."""
        return trg.describe(self.trigger_type)

    @property
    def is_pedestal(self) -> bool:
        """True for pedestal-trigger events (no light, all gains digitised)."""
        return trg.is_pedestal(self.trigger_type)

    @property
    def is_physics(self) -> bool:
        """True for events with a physics (global) trigger, following AKat's ``IsAGlobalTrigger``."""
        return trg.is_physics(self.trigger_type)


_HEADER_FMT = ">bBihiiiiQiHHHHHHHHiQ"   # 64 bytes; optional trailing h (NofHistory)


def decode_header(bank: SFBank) -> Header:
    """Decode the ``Header`` bank of an event into a :class:`Header`."""
    bo = bank.byteorder
    fmt = bo + _HEADER_FMT[1:]
    vals = list(struct.unpack_from(fmt, bank.data, 0))
    nhist = 0
    if bank.size >= struct.calcsize(fmt) + 2:
        nhist = struct.unpack_from(bo + "h", bank.data, struct.calcsize(fmt))[0]
    return Header(*vals, nhist)


@dataclass
class RunHeader:
    """The ``RunHeader`` bank found in the first event of a run (format ``2C2I3S11B``)."""

    major_version: int
    minor_version: int
    run: int
    unix_start_time: int
    shifters: str
    run_type: str      #: e.g. ``'source-60Co'`` or ``'normal'``
    comment: str       #: e.g. ``'+3.50 m'`` (calibration source z position)
    extra: List[int] = field(default_factory=list)

    @property
    def source_z_cm(self) -> Optional[float]:
        """Parse a source position such as ``'+3.50 m'`` from the comment (cm)."""
        s = self.comment.strip().replace(" ", "")
        for unit, scale in (("mm", 0.1), ("cm", 1.0), ("m", 100.0)):
            if s.endswith(unit):
                try:
                    return float(s[:-len(unit)]) * scale
                except ValueError:
                    return None
        return None


def decode_run_header(bank: SFBank) -> RunHeader:
    """Decode the ``RunHeader`` bank (first event of a file) into a :class:`RunHeader`."""
    vals = bank.unpack(unsigned=True)
    strs = [v.decode("latin1", "replace") if isinstance(v, bytes) else str(v) for v in vals[4:7]]
    return RunHeader(vals[0], vals[1], vals[2], vals[3], strs[0], strs[1], strs[2], list(vals[7:]))


@dataclass
class History:
    """The ``History`` bank: trigger records collected within the event window."""

    major_version: int
    minor_version: int
    timestamp: int
    n: int
    trigger_type: np.ndarray
    nsum: np.ndarray
    nsum_5inch: np.ndarray
    nsum_veto_top: np.ndarray
    nsum_veto_upper: np.ndarray
    nsum_veto_lower: np.ndarray
    nsum_veto_bottom: np.ndarray


def decode_history(bank: SFBank) -> History:
    """Decode a ``History`` bank (trigger history words) into a :class:`History`."""
    bo = bank.byteorder
    major, minor, ts, n = struct.unpack_from(bo + "bBQh", bank.data, 0)
    off = 12
    rec = np.dtype([("trig", bo + "i4"), ("nsum", bo + "u2"), ("n5", bo + "u2"),
                    ("top", bo + "u2"), ("upper", bo + "u2"), ("lower", bo + "u2"), ("bottom", bo + "u2")])
    nrec = (bank.size - off) // rec.itemsize
    arr = np.frombuffer(bank.data, dtype=rec, count=nrec, offset=off)
    return History(major, minor, ts, n, arr["trig"].astype(np.int64), arr["nsum"].astype(np.int32),
                   arr["n5"].astype(np.int32), arr["top"].astype(np.int32), arr["upper"].astype(np.int32),
                   arr["lower"].astype(np.int32), arr["bottom"].astype(np.int32))


@dataclass
class HitHeader:
    """``HitHeader`` / ``AntiHitHeader``: which cables fired and how many waveforms each has."""

    major_version: int
    minor_version: int
    nhit: int
    cable: np.ndarray        #: cable numbers (int32[nhit])
    n_waveforms: np.ndarray  #: waveforms per cable (int32[nhit])

    @property
    def total_waveforms(self) -> int:
        """Total number of waveforms announced by the hit header."""
        return int(self.n_waveforms.sum())


def decode_hit_header(bank: SFBank) -> HitHeader:
    """Decode a ``HitHeader``/``AntiHitHeader`` bank into a :class:`HitHeader`."""
    w = bank.array("<u2")
    major, minor = bank.data[0], bank.data[1]
    nhit = int(w[1])
    words = w[2:2 + nhit].astype(np.int32)
    return HitHeader(major, minor, nhit, words & 0xFFF, (words >> 12) & 7)


def event_header(event: SFEvent) -> Optional[Header]:
    """Convenience: the decoded :class:`Header` of *event*."""
    b = event.get("Header")
    return decode_header(b) if b is not None else None
