"""Reader for the KamLAND "Serial Format" (SF / SFZ) raw-data files.

The format is defined by the C++ library in ``SF/`` (O. Tajima, 2001).  A file
is a flat concatenation of *events*; an event is itself an ``SFdata`` record
whose payload is a sequence of named ``SFdata`` records (the *banks*).

Every ``SFdata`` record is laid out as::

    [vsize total] [vsize name_len][name bytes] [vsize form_len][form bytes]
    [data bytes ...] [vsize_r total]

* ``vsize`` is a variable-length big-endian-ish integer: one byte if < 255,
  otherwise ``FF`` + 2 bytes, ``FF FF`` + 3 bytes or ``FF FF FF`` + 4 bytes.
  The length fields include their own size prefix.  The trailing size is
  written byte-reversed so that the file can be read backwards.
* ``form`` is a compressed encoding of the ASCII field-format string
  (e.g. ``2CIB4ILI8BIL``) whose first nibble also carries the byte order of
  the payload (0 = little endian, 1 = big endian).
* The event record's *name* is a 10-byte binary blob: run number (int32),
  sub-run (int16) and event number (int32), big-endian.

The format letters are::

    C 1-byte int   B 2-byte int   I 4-byte int   L 8-byte int
    F 4-byte float D 8-byte float S string (vsize-prefixed)
    R recursive SFdata   A any (opaque)   * repeat-to-end   n(...) groups
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "SFBank", "SFEvent", "SFReader", "vsize_decode", "expand_form",
    "decode_form", "read_record",
]

# Nibble -> format character (SFCodeTable.cc)
_NUM2CHAR = "_CBzIFASLDHRn()*"
_FIXED_SIZE = {"C": 1, "B": 2, "I": 4, "L": 8, "F": 4, "D": 8}
_STRUCT_SIGNED = {"C": "b", "B": "h", "I": "i", "L": "q", "F": "f", "D": "d"}
_STRUCT_UNSIGNED = {"C": "B", "B": "H", "I": "I", "L": "Q", "F": "f", "D": "d"}


def vsize_decode(buf, pos: int = 0) -> Tuple[int, int]:
    """Decode a variable-length size field at ``buf[pos]``.

    Returns ``(value, nbytes_consumed)``.
    """
    b0 = buf[pos]
    if b0 < 255:
        return b0, 1
    b1 = buf[pos + 1]
    if b1 < 255:
        return (b1 << 8) | buf[pos + 2], 3
    b2 = buf[pos + 2]
    if b2 < 255:
        return (b2 << 16) | (buf[pos + 3] << 8) | buf[pos + 4], 5
    return ((buf[pos + 3] << 24) | (buf[pos + 4] << 16)
            | (buf[pos + 5] << 8) | buf[pos + 6]), 7


def vsize_encode(value: int) -> bytes:
    """Inverse of :func:`vsize_decode` (forward form)."""
    if value < 255:
        return bytes([value])
    if value < 65280:
        return bytes([255, value >> 8, value & 255])
    if value < 16711680:
        return bytes([255, 255, value >> 16, (value >> 8) & 255, value & 255])
    return bytes([255, 255, 255, value >> 24, (value >> 16) & 255,
                  (value >> 8) & 255, value & 255])


def expand_form(form: str) -> Tuple[str, int]:
    """Expand a compressed ASCII format such as ``'2CIB4ILI8BIL'``.

    Returns ``(expanded, loop_start)`` where *expanded* is one letter per
    field and *loop_start* is the index from which the pattern repeats until
    the data are exhausted (``-1`` if there is no ``*``).
    Port of ``SFormComp::expand_form``.
    """
    out: List[str] = []
    loop = -1
    number = 0
    i = 0
    n = len(form)
    while i < n:
        ch = form[i]
        if ch.isdigit():
            number = number * 10 + int(ch)
            i += 1
            continue
        if ch == "*":
            loop = len("".join(out))
            number = 1
            i += 1
            continue
        if ch == "(" and number == 0:
            number = 1
        if number > 0:
            if ch != "(":
                out.append(ch * number)
                number = 0
                i += 1
                continue
            depth = 1
            j = i + 1
            while j < n:
                c = form[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            sub, subloop = expand_form(form[i + 1:j])
            if subloop >= 0:
                loop = len("".join(out)) + subloop
            out.append(sub * number)
            number = 0
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), loop


def decode_form(binform: bytes) -> Tuple[str, int]:
    """Decode the binary (nibble-packed) format field.

    Returns ``(ascii_format, endian)`` with endian 0 = little, 1 = big.
    Port of ``SFormComp::bin2asci`` (version 0 encoding only; the
    BWT-compressed variant, version 1, is never used by the KamLAND DAQ).
    """
    if len(binform) == 0:
        return "", 1
    endian = binform[0] >> 4
    if endian >= 4:  # version 1: Burrows-Wheeler compressed format string
        raise NotImplementedError("BWT-compressed SF format strings are not supported")
    out: List[str] = []
    i = 1
    t = _NUM2CHAR[binform[0] & 15]
    if t == "n":
        val, nb = vsize_decode(binform, 1)
        out.append(str(val))
        i = 1 + nb
    elif t != "_":
        out.append(t)
    n = len(binform)
    while i < n:
        c = binform[i]
        t = _NUM2CHAR[c >> 4]
        if t != "_":
            out.append(t)
        t = _NUM2CHAR[c & 15]
        if t == "n":
            val, nb = vsize_decode(binform, i + 1)
            i += nb
            out.append(str(val))
        elif t != "_":
            out.append(t)
        i += 1
    return "".join(out), endian


def read_record(buf, pos: int = 0) -> Tuple[bytes, str, int, memoryview, int]:
    """Parse one ``SFdata`` record starting at ``buf[pos]``.

    Returns ``(name_bytes, ascii_form, endian, data, total_length)``.
    ``data`` is a memoryview into ``buf`` (zero copy).
    """
    mv = memoryview(buf)
    total, ns = vsize_decode(mv, pos)
    p = pos + ns
    nfield, nn = vsize_decode(mv, p)
    name = bytes(mv[p + nn:p + nfield])
    p += nfield
    ffield, fn = vsize_decode(mv, p)
    form, endian = decode_form(bytes(mv[p + fn:p + ffield]))
    p += ffield
    dlen = total - 2 * ns - nfield - ffield
    data = mv[p:p + dlen]
    return name, form, endian, data, total


@dataclass
class SFBank:
    """One named data block inside an event."""

    name: str
    form: str            #: ASCII format, e.g. ``'2CIB4ILI8BIL'``
    endian: int          #: 0 = little endian, 1 = big endian
    data: memoryview     #: raw payload

    @property
    def byteorder(self) -> str:
        """``'<'`` or ``'>'`` from the first nibble of the format string."""
        return ">" if self.endian == 1 else "<"

    @property
    def size(self) -> int:
        """Number of data bytes in the bank."""
        return len(self.data)

    def tobytes(self) -> bytes:
        """The raw payload of the bank."""
        return self.data.tobytes()

    def array(self, dtype) -> np.ndarray:
        """Interpret the whole payload as a flat numpy array of ``dtype``.

        The byte order of the bank is applied automatically for fixed-size
        numeric dtypes given without an explicit byte order.
        """
        dt = np.dtype(dtype)
        if dt.byteorder == "=" and dt.itemsize > 1:
            dt = dt.newbyteorder(self.byteorder)
        n = len(self.data) // dt.itemsize
        return np.frombuffer(self.data, dtype=dt, count=n)

    def unpack(self, fmt: Optional[str] = None, unsigned: bool = False) -> list:
        """Decode the payload according to the SF format string.

        ``fmt`` may be given to override the stored format (e.g. to force
        signed/unsigned interpretation with struct letters).  If ``fmt``
        contains only struct letters it is used verbatim (byte order added).
        Strings (``S``) are returned as ``bytes`` without their trailing NUL;
        recursive records (``R``) are returned as :class:`SFBank`.
        """
        form = self.form if fmt is None else fmt
        expanded, loop = expand_form(form)
        table = _STRUCT_UNSIGNED if unsigned else _STRUCT_SIGNED
        bo = self.byteorder
        mv = self.data
        n = len(mv)
        out: list = []
        pos = 0
        i = 0
        while pos < n:
            if i >= len(expanded):
                if loop < 0:
                    break
                i = loop + (i - loop) % (len(expanded) - loop) if len(expanded) > loop else loop
            ch = expanded[i]
            i += 1
            if ch in _FIXED_SIZE:
                sz = _FIXED_SIZE[ch]
                out.append(struct.unpack_from(bo + table[ch], mv, pos)[0])
                pos += sz
            elif ch in "bBhHiIqQfd":  # raw struct letters
                sz = struct.calcsize(ch)
                out.append(struct.unpack_from(bo + ch, mv, pos)[0])
                pos += sz
            elif ch == "S":
                slen, sn = vsize_decode(mv, pos)
                s = bytes(mv[pos + sn:pos + slen])
                out.append(s.rstrip(b"\0"))
                pos += slen
            elif ch == "R":
                name, form2, endian2, data2, total = read_record(mv, pos)
                out.append(SFBank(name.decode("latin1"), form2, endian2, data2))
                pos += total
            elif ch == "A":
                out.append(bytes(mv[pos:]))
                pos = n
            else:
                raise ValueError(f"unknown format character {ch!r} in {form!r}")
        return out

    def __repr__(self) -> str:
        return f"SFBank({self.name!r}, form={self.form!r}, endian={self.endian}, size={self.size})"


@dataclass
class SFEvent:
    """A decoded event: header identifiers plus a dictionary of banks."""

    run: int
    subrun: int
    event_number: int
    banks: "dict[str, SFBank]" = field(default_factory=dict)
    offset: int = -1     #: byte offset of the event in the file
    size: int = 0        #: total size of the event record in bytes
    index: int = -1      #: sequential index in the file (0-based)

    def __contains__(self, name: str) -> bool:
        return name in self.banks

    def __getitem__(self, name: str) -> SFBank:
        return self.banks[name]

    def get(self, name: str, default=None):
        """The bank named *name*, or *default* if the event does not contain it."""
        return self.banks.get(name, default)

    def names(self) -> List[str]:
        """Names of the banks in the event."""
        return list(self.banks)

    def __repr__(self) -> str:
        return (f"SFEvent(run={self.run}, subrun={self.subrun}, event={self.event_number}, "
                f"banks={self.names()})")


def _event_name_decode(name: bytes) -> Tuple[int, int, int]:
    if len(name) == 10:
        return struct.unpack(">iHi", name)  # big-endian run, subrun, event
    return (-1, -1, -1)


class SFReader:
    """Sequential / random access reader for one SF or SFZ file.

    Example::

        with SFReader(path) as rd:
            for ev in rd:
                hdr = ev["Header"]
                ...

    The reader itself does *not* decompress waveforms; see
    :mod:`claudland.wfcomp` and :class:`claudland.reco.EventReconstructor`.
    """

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = os.fspath(path)
        self._fh = open(self.path, "rb")
        self._fh.seek(0, os.SEEK_END)
        self.file_size = self._fh.tell()
        self._fh.seek(0)
        self._pos = 0
        self._index = 0
        self._offsets: Optional[List[int]] = None

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "SFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying file."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None  # type: ignore[assignment]

    # -- low level ----------------------------------------------------------
    def _read_size_at(self, pos: int) -> Tuple[int, int]:
        self._fh.seek(pos)
        head = self._fh.read(7)
        if len(head) == 0:
            raise EOFError
        return vsize_decode(head, 0)

    def peek_name(self, pos: Optional[int] = None) -> Tuple[int, int, int]:
        """Return ``(run, subrun, event)`` of the event at ``pos`` without
        reading it fully."""
        if pos is None:
            pos = self._pos
        self._fh.seek(pos)
        head = self._fh.read(32)
        if len(head) < 12:
            raise EOFError
        total, ns = vsize_decode(head, 0)
        nfield, nn = vsize_decode(head, ns)
        name = head[ns + nn: ns + nfield]
        return _event_name_decode(bytes(name))

    def read_at(self, pos: int, index: int = -1) -> SFEvent:
        """Read and decode the event record starting at byte ``pos``."""
        total, ns = self._read_size_at(pos)
        self._fh.seek(pos)
        buf = self._fh.read(total)
        if len(buf) < total:
            raise EOFError(f"truncated event at offset {pos}")
        name, form, endian, data, tl = read_record(buf, 0)
        run, subrun, evn = _event_name_decode(name)
        ev = SFEvent(run, subrun, evn, offset=pos, size=total, index=index)
        off = 0
        n = len(data)
        while off < n:
            bname, bform, bendian, bdata, btotal = read_record(data, off)
            ev.banks[bname.decode("latin1")] = SFBank(bname.decode("latin1"), bform, bendian, bdata)
            off += btotal
        self._pos = pos + total
        return ev

    # -- iteration ----------------------------------------------------------
    def rewind(self) -> None:
        """Go back to the first record."""
        self._pos = 0
        self._index = 0

    def next(self) -> Optional[SFEvent]:
        """Read the next event, or return ``None`` at end of file."""
        if self._pos >= self.file_size:
            return None
        try:
            ev = self.read_at(self._pos, self._index)
        except EOFError:
            return None
        self._index += 1
        return ev

    def __iter__(self) -> Iterator[SFEvent]:
        self.rewind()
        while True:
            ev = self.next()
            if ev is None:
                return
            yield ev

    def skip(self, n: int = 1) -> bool:
        """Skip forward ``n`` events without decoding them."""
        for _ in range(n):
            if self._pos >= self.file_size:
                return False
            total, ns = self._read_size_at(self._pos)
            self._pos += total
            self._index += 1
        return True

    # -- random access ------------------------------------------------------
    def build_index(self) -> List[int]:
        """Scan the file once and cache the byte offset of every event."""
        if self._offsets is None:
            offsets = []
            pos = 0
            while pos < self.file_size:
                offsets.append(pos)
                total, ns = self._read_size_at(pos)
                pos += total
            self._offsets = offsets
        return self._offsets

    def __len__(self) -> int:
        return len(self.build_index())

    def __getitem__(self, i: int) -> SFEvent:
        offsets = self.build_index()
        if i < 0:
            i += len(offsets)
        return self.read_at(offsets[i], i)

    def events(self, start: int = 0, stop: Optional[int] = None) -> Iterator[SFEvent]:
        """Iterate over events ``start:stop`` by sequential index."""
        offsets = self.build_index()
        stop = len(offsets) if stop is None else min(stop, len(offsets))
        for i in range(start, stop):
            yield self.read_at(offsets[i], i)
