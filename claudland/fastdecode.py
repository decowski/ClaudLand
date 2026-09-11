"""ctypes bridge to the C waveform decoder (:file:`_cdecode.c`).

The shared library is compiled on first import with the system C compiler
(``cc -O2 -shared -fPIC``) into the package directory and reused afterwards
(rebuilt when the ``.c`` file is newer).  If no compiler is available, or
``CLAUDLAND_NO_CDECODE=1`` is set, :data:`available` is ``False`` and
:class:`~claudland.wfcomp.WaveformDecompressor` falls back to the pure-Python
decoder, which is ~20x slower but produces identical output.

Run ``python3 -m claudland.fastdecode`` to (re)build and report the status.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from . import huffman_tables

__all__ = ["available", "decode_cmp", "decode_atwd", "build", "library_path", "NSAMPLES"]

NSAMPLES = 128
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "_cdecode.c"
_LIB = _HERE / ("_cdecode" + (".dylib" if sys.platform == "darwin" else ".so"))

_ARR = {}


def _arrays():
    """int32 copies of the private Huffman tables (loaded on first use)."""
    if not _ARR:
        h = huffman_tables.load()
        for k in ("TABLE", "TREE", "NBITS"):
            _ARR[k] = np.ascontiguousarray(h[k], dtype=np.int32)
    return _ARR["TABLE"], _ARR["TREE"], _ARR["NBITS"]

_lib: Optional[ctypes.CDLL] = None
available = False
build_error: Optional[str] = None


def library_path() -> Path:
    """Path of the compiled shared library."""
    return _LIB


def build(force: bool = False, verbose: bool = False) -> bool:
    """Compile ``_cdecode.c`` if needed.  Returns True when the library is usable."""
    global build_error
    if not force and _LIB.exists() and _LIB.stat().st_mtime >= _SRC.stat().st_mtime:
        return True
    cc = os.environ.get("CC", "cc")
    cmd = [cc, "-O2", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:   # compiler missing
        build_error = f"{cc}: {exc}"
        return False
    if r.returncode != 0:
        build_error = r.stderr.strip() or f"{cc} exited with {r.returncode}"
        if verbose:
            print(build_error, file=sys.stderr)
        return False
    return True


def _load() -> Optional[ctypes.CDLL]:
    global build_error
    if os.environ.get("CLAUDLAND_NO_CDECODE"):
        build_error = "disabled by CLAUDLAND_NO_CDECODE"
        return None
    if not build():
        return None
    try:
        lib = ctypes.CDLL(str(_LIB))
    except OSError as exc:
        build_error = str(exc)
        return None
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    i16p = ctypes.POINTER(ctypes.c_int16)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    L = ctypes.c_long
    lib.kl_decode_cmp.restype = L
    lib.kl_decode_cmp.argtypes = [u8p, L, u16p, L, u8p, i64p, L, u16p, L, i32p, i32p, i32p, i32p, i16p, L]
    lib.kl_decode_atwd.restype = L
    lib.kl_decode_atwd.argtypes = [u32p, L, u16p, L, u8p, i32p, i16p, L]
    return lib


_ERRORS = {-1: "bank truncated", -2: "output buffer too small", -3: "corrupt Huffman block",
           -4: "cable outside connection table"}


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def _want(gains: Sequence[int]) -> np.ndarray:
    w = np.zeros(4, dtype=np.uint8)
    w[list(gains)] = 1
    return w


def _count(words: np.ndarray) -> int:
    return int(((words.astype(np.int64) >> 12) & 7).sum())


def decode_cmp(cmp: bytes, words: np.ndarray, gains: Sequence[int],
               base_row: np.ndarray, ped_table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a ``CmpATWD`` payload.

    Returns ``(meta int32[n,5], samples int16[n,128])`` with meta columns
    cable, atwd, gain, launch_offset, flags (bit 0 = Huffman bit count outside
    the padded word, bit 1 = block stored raw in Thorsten format).
    """
    if _lib is None:
        raise RuntimeError("C decoder not available")
    buf = np.frombuffer(cmp, dtype=np.uint8)
    words = np.ascontiguousarray(words, dtype=np.uint16)
    base = np.ascontiguousarray(base_row, dtype=np.int64)
    ped = np.ascontiguousarray(ped_table, dtype=np.uint16)
    n_max = _count(words)
    meta = np.empty((n_max, 5), dtype=np.int32)
    samples = np.empty((n_max, NSAMPLES), dtype=np.int16)
    TABLE, TREE, NBITS = _arrays()
    n = _lib.kl_decode_cmp(_ptr(buf, ctypes.c_uint8), len(buf), _ptr(words, ctypes.c_uint16), len(words),
                           _ptr(_want(gains), ctypes.c_uint8), _ptr(base, ctypes.c_int64), len(base),
                           _ptr(ped, ctypes.c_uint16), ped.shape[0],
                           _ptr(TABLE, ctypes.c_int32), _ptr(TREE, ctypes.c_int32), _ptr(NBITS, ctypes.c_int32),
                           _ptr(meta, ctypes.c_int32), _ptr(samples, ctypes.c_int16), n_max)
    if n < 0:
        raise ValueError(f"CmpATWD decode failed: {_ERRORS.get(n, n)}")
    return meta[:n], samples[:n]


def decode_atwd(data: np.ndarray, words: np.ndarray, gains: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Decode an uncompressed ``ATWD`` bank given as a ``uint32`` array."""
    if _lib is None:
        raise RuntimeError("C decoder not available")
    data = np.ascontiguousarray(data, dtype=np.uint32)
    words = np.ascontiguousarray(words, dtype=np.uint16)
    n_max = _count(words)
    meta = np.empty((n_max, 5), dtype=np.int32)
    samples = np.empty((n_max, NSAMPLES), dtype=np.int16)
    n = _lib.kl_decode_atwd(_ptr(data, ctypes.c_uint32), len(data), _ptr(words, ctypes.c_uint16), len(words),
                            _ptr(_want(gains), ctypes.c_uint8),
                            _ptr(meta, ctypes.c_int32), _ptr(samples, ctypes.c_int16), n_max)
    if n < 0:
        raise ValueError(f"ATWD decode failed: {_ERRORS.get(n, n)}")
    return meta[:n], samples[:n]


_lib = _load()
available = _lib is not None


if __name__ == "__main__":
    ok = build(force="--force" in sys.argv, verbose=True)
    print(f"library: {_LIB}")
    print(f"built: {ok}" + ("" if ok else f" ({build_error})"))
    print(f"available in this process: {available}" + ("" if available else f" ({build_error})"))
