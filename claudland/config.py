"""Locations of the KamLAND-private inputs (``claudland.toml``).

The package itself carries no collaboration data.  The PMT coordinate table,
the run list, the Huffman tables of the waveform compression, the raw-data
directory and the calibration constants are all looked up through a small
TOML file, found in this order:

1. the file named by the environment variable ``CLAUDLAND_CONFIG``;
2. ``claudland.toml`` in the current directory or one of its parents;
3. ``claudland.toml`` next to the package (the repository root);
4. ``~/.config/claudland/claudland.toml``.

Relative paths inside the file are resolved from the directory holding it.
Missing entries fall back to :data:`DEFAULTS`.  Accessing an input that does
not exist on disk raises :class:`PrivateInputMissing` with a message saying
which file to obtain and where it is expected.

Example::

    from claudland import config
    cfg = config.get()
    pmts = PMTTable.load()                 # uses cfg.pmt_table
    path = cfg.find_run(2283)              # .../run_002283_000000_000001.sfz
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

__all__ = ["Config", "PrivateInputMissing", "get", "load", "reset", "FILENAME", "DEFAULTS"]

FILENAME = "claudland.toml"
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: values used when the file or an entry is missing (relative to the repository root)
DEFAULTS: Dict[str, str] = {
    "private_dir": "private",
    "pmt_table": "private/pmt_xyz.dat",
    "run_info": "private/run-info.table",
    "huffman_tables": "private/huffman_tables.json",
    "data_dir": "../Run",
    "cache_dir": "cache",
    "tq": "cache/calib_center.json",
    "time_pdf": "cache/timepdf.npz",
}


class PrivateInputMissing(FileNotFoundError):
    """A KamLAND-private input file is not available at the configured location."""


@dataclass
class Config:
    """Resolved configuration; all path attributes are absolute :class:`~pathlib.Path` objects."""

    source: Optional[Path]          #: the TOML file that was read (None: built-in defaults)
    base: Path                      #: directory relative paths were resolved from
    private_dir: Path
    pmt_table: Path
    run_info: Path
    huffman_tables: Path
    data_dir: Path
    cache_dir: Path
    tq: Path                        #: default TQ calibration JSON
    time_pdf: Path                  #: default time-PDF file
    extra: Dict[str, str] = field(default_factory=dict)   #: any further keys of the file

    _DESCRIPTIONS = {
        "pmt_table": "PMT coordinate table (cable x y z in cm; Kat/src/pmt_xyz.cc)",
        "run_info": "run list (run-info.table)",
        "huffman_tables": "Huffman tables of the .sfz waveform compression (from WFComp/trees.hh)",
        "data_dir": "directory with the raw .sf/.sfz run files",
        "tq": "TQ calibration JSON (made by scripts/zscan_calibrate.py)",
        "time_pdf": "time-PDF file (made by scripts/zscan_calibrate.py)",
    }

    def require(self, key: str) -> Path:
        """Return the path for *key*, raising :class:`PrivateInputMissing` if it does not exist."""
        p = getattr(self, key)
        if not Path(p).exists():
            what = self._DESCRIPTIONS.get(key, key)
            where = f"configured in {self.source}" if self.source else f"built-in default (no {FILENAME} found)"
            raise PrivateInputMissing(f"{what} not found at {p} [{where}]. "
                                      f"This input is private to the KamLAND collaboration; "
                                      f"place it there or point '{key}' in {FILENAME} to it.")
        return Path(p)

    def find_run(self, run: int, pattern: str = "run_{run:06d}_*.sf*") -> Optional[Path]:
        """First raw-data file of *run* below :attr:`data_dir` (searched recursively), or ``None``."""
        if not self.data_dir.exists():
            return None
        hits = sorted(self.data_dir.rglob(pattern.format(run=run)))
        return hits[0] if hits else None

    def optional(self, key: str) -> Optional[Path]:
        """The path for *key* if the file exists, else ``None`` (for script defaults)."""
        p = Path(getattr(self, key))
        return p if p.exists() else None


def _candidates() -> list:
    out = []
    env = os.environ.get("CLAUDLAND_CONFIG")
    if env:
        out.append(Path(env).expanduser())
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        out.append(d / FILENAME)
    out.append(_REPO_ROOT / FILENAME)
    out.append(Path.home() / ".config" / "claudland" / FILENAME)
    return out


def load(path: Optional[os.PathLike] = None) -> Config:
    """Read the configuration from *path* (default: the search order of the module docstring)."""
    src = None
    if path is not None:
        src = Path(path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"configuration file {src} not found")
    else:
        for c in _candidates():
            if c.is_file():
                src = c
                break
    values: Dict[str, str] = {}
    if src is not None:
        import tomllib
        with open(src, "rb") as fh:
            doc = tomllib.load(fh)
        for section in doc.values():
            if isinstance(section, dict):
                values.update({k: str(v) for k, v in section.items()})
        base = src.resolve().parent
    else:
        base = _REPO_ROOT
    merged = dict(DEFAULTS)
    merged.update(values)

    def resolve(key: str) -> Path:
        p = Path(merged[key]).expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    known = set(DEFAULTS)
    return Config(source=src, base=base, private_dir=resolve("private_dir"), pmt_table=resolve("pmt_table"),
                  run_info=resolve("run_info"), huffman_tables=resolve("huffman_tables"), data_dir=resolve("data_dir"),
                  cache_dir=resolve("cache_dir"), tq=resolve("tq"), time_pdf=resolve("time_pdf"),
                  extra={k: v for k, v in merged.items() if k not in known})


_CACHED: Optional[Config] = None


def get() -> Config:
    """The process-wide configuration (loaded on first use; see :func:`reset`)."""
    global _CACHED
    if _CACHED is None:
        _CACHED = load()
    return _CACHED


def reset(path: Optional[os.PathLike] = None) -> Config:
    """Reload the configuration (optionally from an explicit *path*) and make it the process default."""
    global _CACHED
    _CACHED = load(path)
    return _CACHED
