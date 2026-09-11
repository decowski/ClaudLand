"""Decompression of KamLAND ATWD waveforms stored in ``.sfz`` files.

The ``sfzip`` tool (``SF/SFWaveformCoDec.cc`` + ``WFComp/``) replaced the raw
``ATWD`` / ``AntiATWD`` banks by ``CmpATWD`` / ``CmpAntiATWD`` banks and added
two constant banks to the first event of each file:

``ConnectionTable``
    ``uint16 ncable`` followed by one ``uint16`` per cable number:
    ``crate = (w >> 9) & 0xf``, ``slot = (w >> 4) & 0x1f``, ``channel = w & 0xf``.
``CmpPedestal``
    ``19200 x 128 uint16`` pedestal waveforms indexed by
    ``(crate-1, slot-2, channel, atwd, gain)`` -- these are the pedestals that
    were *subtracted before compression* and must be added back.

Each compressed waveform block (``SFCD::CmpATWD``) is::

    u8 size            total block size including this 4-byte header
    u8 launch_offset   bit 7 = sign, bits 0-6 = magnitude (units of 25 ns)
    u8 flags           bits 0-1 = gain (0 H, 1 M, 2 L, 3 clock), bit 2 = ATWD (0 A, 1 B)
    u8 comp_info       unused
    u16 nbits          little endian, number of code bits that follow
    ...                ceil(nbits/8) bytes of code, or 43 x u32 raw samples
                       when nbits == 1376 ("Thorsten format", 3 x 10-bit
                       samples per little-endian 32-bit word)

The Huffman code (``TDiffEntropyHuffmanCoDec``) encodes 128 10-bit symbols
produced by: pedestal subtraction, a backwards difference transform with a
sign "reflect" trick and a power-compression shift (M. Batygov).  The
decoder below is a faithful port of ``dec_d`` / ``Decode`` in
``WFComp/DiffEntropyHuffmanCoDec.cc`` and of the ``Bitter`` bit reader.

Bits are consumed MSB-first from consecutive little-endian 32-bit words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import huffman_tables

__all__ = [
    "ConnectionTable", "CompressionPedestals", "HuffmanWaveformDecoder",
    "decode_thorsten", "CompressedWaveform", "iter_compressed_blocks",
    "WaveformDecompressor", "WaveformBatch", "NSAMPLES", "THORSTEN_BITS",
]

NSAMPLES = 128
THORSTEN_BITS = 43 * 32       # 1376: marks an uncompressed block
DSTART = 18                   # seed of the difference transform (DiffEntropyHuffmanCoDec.cc)

# Pedestal array geometry (TKLCompressionPedestalManager.hh)
N_CRATES, FIRST_CRATE = 10, 1
N_SLOTS, FIRST_SLOT = 20, 2
N_CHANNELS = 12
N_ATWDS = 2
N_GAINS = 4
N_PEDESTALS = N_CRATES * N_SLOTS * N_CHANNELS * N_ATWDS * N_GAINS  # 19200


def pedestal_index(crate: int, slot: int, channel: int, atwd: int, gain: int) -> int:
    """Row index into the ``(19200, 128)`` compression-pedestal array."""
    return ((((crate - FIRST_CRATE) * N_SLOTS + (slot - FIRST_SLOT)) * N_CHANNELS
             + channel) * N_ATWDS + atwd) * N_GAINS + gain


class ConnectionTable:
    """Cable number -> (crate, slot, channel) map from the ``ConnectionTable`` bank."""

    def __init__(self, words: np.ndarray):
        words = np.asarray(words, dtype=np.uint16)
        self.n = int(words[0])
        w = words[1:1 + self.n].astype(np.int32)
        self.crate = (w >> 9) & 0xF
        self.slot = (w >> 4) & 0x1F
        self.channel = w & 0xF
        # precomputed base row (gain 0, atwd 0) of the pedestal array per cable
        valid = (self.crate >= FIRST_CRATE) & (self.crate < FIRST_CRATE + N_CRATES) & \
                (self.slot >= FIRST_SLOT) & (self.slot < FIRST_SLOT + N_SLOTS)
        self.valid = valid
        base = (((self.crate - FIRST_CRATE) * N_SLOTS + (self.slot - FIRST_SLOT)) * N_CHANNELS
                + self.channel) * N_ATWDS * N_GAINS
        self.base_row = np.where(valid, base, -1)

    @classmethod
    def from_bank(cls, bank) -> "ConnectionTable":
        """Build from the corresponding SF bank."""
        return cls(bank.array("<u2"))

    def __len__(self) -> int:
        return self.n

    def lookup(self, cable: int) -> Tuple[int, int, int]:
        """(crate, slot, channel) of a cable."""
        return int(self.crate[cable]), int(self.slot[cable]), int(self.channel[cable])

    def pedestal_row(self, cable: int, atwd: int, gain: int) -> int:
        """Row of the ``CmpPedestal`` table for (cable, ATWD, gain), or -1 if not in the front-end map."""
        b = int(self.base_row[cable])
        if b < 0:
            return -1
        return b + atwd * N_GAINS + gain


class CompressionPedestals:
    """The ``CmpPedestal`` bank as a ``(19200, 128)`` uint16 array."""

    def __init__(self, table: np.ndarray):
        table = np.asarray(table, dtype=np.uint16)
        if table.size != N_PEDESTALS * NSAMPLES:
            raise ValueError(f"CmpPedestal has {table.size} values, expected {N_PEDESTALS * NSAMPLES}")
        self.table = table.reshape(N_PEDESTALS, NSAMPLES)

    @classmethod
    def from_bank(cls, bank) -> "CompressionPedestals":
        """Build from the corresponding SF bank."""
        return cls(bank.array("<u2"))

    def __getitem__(self, row: int) -> np.ndarray:
        return self.table[row]


# ---------------------------------------------------------------------------
# Bit-level Huffman decoding
# ---------------------------------------------------------------------------
def _tables():
    """(TABLE, TREE, NBITS) of the private Huffman tables, loaded on first use."""
    h = huffman_tables.load()
    return h["TABLE"], h["TREE"], h["NBITS"]


class HuffmanWaveformDecoder:
    """Stateless decoder for one compressed 128-sample block."""

    @staticmethod
    def _bitstream(code: bytes, nbits: int) -> int:
        """Pack the little-endian u32 words into one big integer (MSB first)."""
        nwords = (nbits + 31) // 32
        need = nwords * 4
        if len(code) < need:
            code = code + b"\0" * (need - len(code))
        words = np.frombuffer(code, dtype="<u4", count=nwords)
        # big-endian byte string of the words == MSB-first bit order used by Bitter
        return int.from_bytes(words.astype(">u4").tobytes(), "big"), nwords * 32

    @classmethod
    def decode_symbols(cls, code: bytes, nbits: int) -> Tuple[List[int], int]:
        """Return the 128 transformed symbols and the number of bits consumed."""
        stream, total = cls._bitstream(code, nbits)
        out = [0] * NSAMPLES
        pos = 0  # bits consumed
        table, tree, nb = _tables()
        for i in range(NSAMPLES):
            shift = total - pos - 12
            t = (stream >> shift) & 0xFFF if shift >= 0 else (stream << -shift) & 0xFFF
            v = table[t]
            if v >= 2048:
                sym = v - 2048
                pos += nb[sym]
            else:
                ptr = v
                pos += 12
                while True:
                    bit = (stream >> (total - pos - 1)) & 1
                    pos += 1
                    node = tree[ptr + bit]
                    if node >= 2048:
                        sym = node - 2048
                        break
                    ptr = node
            out[i] = sym
        return out, pos

    @staticmethod
    def inverse_transform(darray: Sequence[int]) -> np.ndarray:
        """Port of ``dec_d`` (POWC + reflect undo, then difference decode)."""
        d = list(darray)
        # undo power-compression shift
        dsav = 0
        for m in range(127, -1, -1):
            if ((dsav + 2) & 1023) > 4:
                shift = dsav + 2 if (dsav & 512) else dsav - 2
                d[m] = (d[m] + shift) & 1023
            dsav = d[m]
        # undo reflection
        dold = 0
        for m in range(127, -1, -1):
            if (dold & 512) == 0:
                d[m] = (-d[m]) & 1023
            dold = d[m]
        # difference decode (no masking here; done after pedestal restore)
        s = np.empty(NSAMPLES, dtype=np.int64)
        dold = DSTART
        for m in range(127, -1, -1):
            dold = dold + d[m]
            s[m] = dold
        return s

    @classmethod
    def decode(cls, code: bytes, nbits: int, pedestal: np.ndarray) -> np.ndarray:
        """Full decode: Huffman -> inverse transform -> add pedestal, mask to 10 bits."""
        syms, used = cls.decode_symbols(code, nbits)
        wave = cls.inverse_transform(syms)
        wave = (wave + pedestal.astype(np.int64)) & 1023
        return wave.astype(np.int16)


def decode_thorsten(raw: bytes) -> np.ndarray:
    """Unpack 43 little-endian u32 words holding 128 10-bit samples.

    Word ``w`` (0..42) holds samples ``3*(42-w) + k`` in bits ``10*k``.
    """
    words = np.frombuffer(raw, dtype="<u4", count=43).astype(np.int64)
    w = words[::-1]  # word 42 first -> samples 0,1,2,...
    s = np.empty(129, dtype=np.int64)
    s[0::3] = w & 0x3FF
    s[1::3] = (w >> 10) & 0x3FF
    s[2::3] = (w >> 20) & 0x3FF
    return s[:NSAMPLES].astype(np.int16)


@dataclass
class CompressedWaveform:
    """One compressed (or raw Thorsten) block of a ``CmpATWD`` bank before decoding."""
    cable: int
    atwd: int          #: 0 = A, 1 = B
    gain: int          #: 0 high, 1 medium, 2 low, 3 clock
    launch_offset: int #: signed, units of 25 ns
    nbits: int
    payload: bytes     #: code bits (or raw Thorsten words)
    comp_info: int = 0

    @property
    def is_raw(self) -> bool:
        """True if the block holds an uncompressed Thorsten-format waveform."""
        return self.nbits == THORSTEN_BITS


def _decode_launch(byte: int) -> int:
    return -(byte & 0x7F) if (byte & 0x80) else byte


def iter_compressed_blocks(hitheader_words: np.ndarray, cmp: bytes) -> Iterator[CompressedWaveform]:
    """Walk a ``CmpATWD`` payload in the order given by the ``HitHeader`` bank.

    ``hitheader_words`` are the per-hit ``uint16`` words
    (``cable = w & 0xfff``, ``n_waveforms = (w >> 12) & 7``).
    """
    pos = 0
    n = len(cmp)
    for w in hitheader_words:
        w = int(w)
        cable = w & 0xFFF
        nwf = (w >> 12) & 7
        for _ in range(nwf):
            if pos + 6 > n:
                raise ValueError("CmpATWD bank truncated")
            size = cmp[pos]
            launch = _decode_launch(cmp[pos + 1])
            flags = cmp[pos + 2]
            info = cmp[pos + 3]
            nbits = cmp[pos + 4] | (cmp[pos + 5] << 8)
            payload = cmp[pos + 6:pos + size]
            yield CompressedWaveform(cable, (flags >> 2) & 1, flags & 3, launch, nbits, payload, info)
            pos += size


@dataclass
class Waveform:
    """One decoded ATWD waveform (raw ADC counts, pedestal *not* subtracted)."""

    cable: int
    atwd: int
    gain: int
    launch_offset: int
    samples: np.ndarray  #: int16[128], 0..1023

    @property
    def gain_name(self) -> str:
        """``H``, ``M``, ``L`` or ``C`` (clock)."""
        return "HMLC"[self.gain]


@dataclass
class WaveformBatch:
    """All decoded waveforms of one bank as flat arrays (the fast-path container).

    Iterating yields :class:`Waveform` objects, so a batch can be used wherever
    a list of waveforms is expected.
    """

    cable: np.ndarray    #: int32[n]
    atwd: np.ndarray     #: int32[n]
    gain: np.ndarray     #: int32[n]
    launch: np.ndarray   #: int32[n] launch offset (units of 25 ns)
    samples: np.ndarray  #: int16[n, 128]

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __iter__(self) -> Iterator[Waveform]:
        for i in range(len(self)):
            yield Waveform(int(self.cable[i]), int(self.atwd[i]), int(self.gain[i]),
                           int(self.launch[i]), self.samples[i])

    def select(self, mask: np.ndarray) -> "WaveformBatch":
        """Sub-batch of the waveforms selected by *mask*."""
        return WaveformBatch(self.cable[mask], self.atwd[mask], self.gain[mask], self.launch[mask], self.samples[mask])

    @classmethod
    def empty(cls) -> "WaveformBatch":
        """A batch with no waveforms."""
        z = np.zeros(0, dtype=np.int32)
        return cls(z, z.copy(), z.copy(), z.copy(), np.zeros((0, NSAMPLES), dtype=np.int16))

    @classmethod
    def from_meta(cls, meta: np.ndarray, samples: np.ndarray) -> "WaveformBatch":
        """Build from the decoder output (meta columns cable, atwd, gain, launch, flags)."""
        return cls(meta[:, 0].copy(), meta[:, 1].copy(), meta[:, 2].copy(), meta[:, 3].copy(), samples)

    @classmethod
    def from_list(cls, wfs: Sequence[Waveform]) -> "WaveformBatch":
        """Build from a list of :class:`Waveform` (a batch is returned unchanged)."""
        if isinstance(wfs, WaveformBatch):
            return wfs
        wfs = list(wfs)
        if not wfs:
            return cls.empty()
        return cls(np.array([w.cable for w in wfs], dtype=np.int32), np.array([w.atwd for w in wfs], dtype=np.int32),
                   np.array([w.gain for w in wfs], dtype=np.int32), np.array([w.launch_offset for w in wfs], dtype=np.int32),
                   np.stack([w.samples for w in wfs]).astype(np.int16))


class WaveformDecompressor:
    """Decompress ``CmpATWD``-style banks for a whole file.

    Feed it the first event (which carries ``ConnectionTable`` and
    ``CmpPedestal``) via :meth:`load_constants`; afterwards
    :meth:`decompress` returns the list of :class:`Waveform` for any event and
    :meth:`decompress_arrays` the same data as a :class:`WaveformBatch`.

    When the C decoder (:mod:`claudland.fastdecode`) is available it is used for
    both; ``use_c=False`` forces the pure-Python decoder.
    """

    ID_BANKS = ("HitHeader", "CmpATWD")
    OD_BANKS = ("AntiHitHeader", "CmpAntiATWD")
    RAW_BANKS = {"ID": "ATWD", "OD": "AntiATWD"}

    def __init__(self, use_c: Optional[bool] = None):
        self.connection: Optional[ConnectionTable] = None
        self.pedestals: Optional[CompressionPedestals] = None
        self.n_raw = 0
        self.n_compressed = 0
        self.n_bit_mismatch = 0
        from . import fastdecode
        self._fast = fastdecode if (fastdecode.available if use_c is None else use_c) else None
        if use_c and not fastdecode.available:
            raise RuntimeError(f"C decoder requested but not available: {fastdecode.build_error}")

    @property
    def uses_c(self) -> bool:
        """True if the C decoder is in use."""
        return self._fast is not None

    @property
    def ready(self) -> bool:
        """True once the compression constants of the file were loaded."""
        return self.connection is not None and self.pedestals is not None

    def load_constants(self, event) -> bool:
        """Pick up ``ConnectionTable`` / ``CmpPedestal`` if present in *event*."""
        found = False
        if "ConnectionTable" in event:
            self.connection = ConnectionTable.from_bank(event["ConnectionTable"])
            found = True
        if "CmpPedestal" in event:
            self.pedestals = CompressionPedestals.from_bank(event["CmpPedestal"])
            found = True
        return found

    def decode_block(self, blk: CompressedWaveform) -> np.ndarray:
        """Decode one :class:`CompressedWaveform` to 128 ADC samples (pedestal restored)."""
        if blk.is_raw:
            self.n_raw += 1
            return decode_thorsten(blk.payload)
        if not self.ready:
            raise RuntimeError("compression constants not loaded (call load_constants on the first event)")
        row = self.connection.pedestal_row(blk.cable, blk.atwd, blk.gain)
        ped = self.pedestals[row] if row >= 0 else np.zeros(NSAMPLES, np.uint16)
        syms, used = HuffmanWaveformDecoder.decode_symbols(blk.payload, blk.nbits)
        if not (blk.nbits - 32 < used <= blk.nbits):   # nbits is padded to whole 32-bit words
            self.n_bit_mismatch += 1
        wave = HuffmanWaveformDecoder.inverse_transform(syms)
        self.n_compressed += 1
        return ((wave + ped.astype(np.int64)) & 1023).astype(np.int16)

    def decompress_banks(self, hitheader_bank, cmp_bank, gains=(0, 1, 2, 3)) -> List[Waveform]:
        """Decode the waveforms of a hit-header / compressed-waveform bank pair (pure Python)."""
        hh = hitheader_bank.array("<u2")   # C C then u16 words
        nhit = int(hh[1])
        words = hh[2:2 + nhit]
        cmp = cmp_bank.tobytes()
        out: List[Waveform] = []
        for blk in iter_compressed_blocks(words, cmp):
            if blk.gain in gains:
                out.append(Waveform(blk.cable, blk.atwd, blk.gain, blk.launch_offset, self.decode_block(blk)))
        return out

    def _banks(self, event, detector: str):
        det = detector.upper()
        hname, cname = self.ID_BANKS if det == "ID" else self.OD_BANKS
        if hname not in event:
            return None, None, None
        if cname in event:
            return event[hname], event[cname], None
        rawname = self.RAW_BANKS[det]
        if rawname in event:
            return event[hname], None, event[rawname]
        return None, None, None

    def decompress_arrays(self, event, detector: str = "ID", gains=(0, 1, 2, 3)) -> WaveformBatch:
        """Like :meth:`decompress` but returns a :class:`WaveformBatch` (fast path)."""
        hbank, cbank, rbank = self._banks(event, detector)
        if hbank is None:
            return WaveformBatch.empty()
        if self._fast is None:
            return WaveformBatch.from_list(self.decompress(event, detector, gains))
        hh = hbank.array("<u2")
        words = hh[2:2 + int(hh[1])]
        if cbank is not None:
            if not self.ready:
                raise RuntimeError("compression constants not loaded (call load_constants on the first event)")
            meta, samples = self._fast.decode_cmp(cbank.tobytes(), words, gains,
                                                  self.connection.base_row, self.pedestals.table)
            n_raw = int((meta[:, 4] & 2).astype(bool).sum())
            self.n_bit_mismatch += int((meta[:, 4] & 1).sum())
            self.n_raw += n_raw
            self.n_compressed += len(meta) - n_raw
        else:
            meta, samples = self._fast.decode_atwd(rbank.array("<u4"), words, gains)
        return WaveformBatch.from_meta(meta, samples)

    def decompress(self, event, detector: str = "ID", gains=(0, 1, 2, 3)) -> List[Waveform]:
        """Return the waveforms of the inner (``"ID"``) or outer (``"OD"``) detector.

        ``gains`` selects which gain channels to decode (0 high, 1 medium,
        2 low, 3 clock).  Works transparently for uncompressed ``.sf`` events
        too (``ATWD`` banks).
        """
        if self._fast is not None:
            return list(self.decompress_arrays(event, detector, gains))
        hbank, cbank, rbank = self._banks(event, detector)
        if hbank is None:
            return []
        if cbank is not None:
            return self.decompress_banks(hbank, cbank, gains)
        return [w for w in decode_atwd_bank(hbank, rbank) if w.gain in gains]


def decode_atwd_bank(hitheader_bank, atwd_bank) -> List[Waveform]:
    """Decode an *uncompressed* ``ATWD`` bank (44 x u32 per waveform).

    Layout per waveform (SFATWDBank.cc): word 0 = ``atwd<<30 | gain<<28 |
    launch<<20 | s[127]<<10 | s[126]``; words 1..42 hold samples
    ``125-3k, 124-3k, 123-3k`` in bits 20, 10, 0.
    """
    hh = hitheader_bank.array("<u2")
    nhit = int(hh[1])
    words = hh[2:2 + nhit]
    data = atwd_bank.array("<u4").astype(np.int64)
    out: List[Waveform] = []
    idx = 0
    for w in words:
        w = int(w)
        cable = w & 0xFFF
        nwf = (w >> 12) & 7
        for _ in range(nwf):
            blk = data[idx:idx + 43]
            idx += 43
            head = int(blk[0])
            atwd = (head >> 30) & 1
            gain = (head >> 28) & 3
            launch = _decode_launch((head >> 20) & 0xFF)
            s = np.empty(NSAMPLES, dtype=np.int16)
            s[127] = (head >> 10) & 0x3FF
            s[126] = head & 0x3FF
            body = blk[1:43]
            k = np.arange(42)
            s[125 - 3 * k] = (body >> 20) & 0x3FF
            s[124 - 3 * k] = (body >> 10) & 0x3FF
            s[123 - 3 * k] = body & 0x3FF
            out.append(Waveform(cable, atwd, gain, launch, s))
    return out
