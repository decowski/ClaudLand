"""The KamLAND run list (``run-info.table``).

One line per physics run::

    000166   2002/03/05(Tue)02:22(57)   5.764   1015262577  0   1   OD inefficiency ~13%

The columns are, as far as they could be established from the file itself:

1. run number;
2. start time in JST, ``YYYY/MM/DD(Dow)HH:MM(SS)`` with the seconds in
   parentheses;
3. duration in hours;
4. start time as Unix seconds;
5. a 0/1 flag whose meaning is not documented (0 before run 192, for stub
   entries filled automatically and again from run 16320 / July 2020 on;
   most likely a bookkeeping or verification flag);
6. run grade: 0 good, 1 and 5/6 partially bad, 9/10 bad;
7. a free-text comment, which for partially bad runs often contains veto
   intervals as 40 MHz time stamps, e.g. ``veto(141054016614 ~ end)``.

The file is private to the collaboration and is read from the location given
by ``run_info`` in ``claudland.toml`` (:mod:`claudland.config`).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["RUN_DTYPE", "load", "lookup", "veto_intervals", "good_runs"]

RUN_DTYPE = np.dtype([
    ("run", np.int32), ("start", "U26"), ("hours", np.float64), ("unix_start", np.int64),
    ("flag", np.int8), ("grade", np.int8), ("comment", "U200"),
])

_LINE = re.compile(r"^\s*(\d+)\s+(\S+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s*(.*?)\s*$")
_VETO = re.compile(r"veto\(\s*(\d+|start|begin)\s*~\s*(\d+|end)\s*\)", re.I)

_CACHE: Optional[np.ndarray] = None


def load(path=None) -> np.ndarray:
    """The whole table as a ``RUN_DTYPE`` structured array (cached when read from the configuration)."""
    global _CACHE
    if path is None and _CACHE is not None:
        return _CACHE
    from . import config
    p = config.get().require("run_info") if path is None else path
    rows = []
    with open(p, errors="replace") as fh:
        for line in fh:
            m = _LINE.match(line)
            if not m:
                continue
            rows.append((int(m.group(1)), m.group(2), float(m.group(3)), int(m.group(4)),
                         int(m.group(5)), int(m.group(6)), m.group(7)[:200]))
    tab = np.array(rows, dtype=RUN_DTYPE)
    if path is None:
        _CACHE = tab
    return tab


def lookup(run: int) -> Optional[np.void]:
    """The table row of *run*, or ``None`` if the run is not listed (calibration runs are not)."""
    tab = load()
    i = np.flatnonzero(tab["run"] == run)
    return tab[i[0]] if len(i) else None


def good_runs(max_grade: int = 0) -> np.ndarray:
    """Run numbers with ``grade <= max_grade``."""
    tab = load()
    return tab["run"][tab["grade"] <= max_grade]


def veto_intervals(comment: str) -> List[Tuple[Optional[int], Optional[int]]]:
    """Veto intervals ``(start, end)`` in 40 MHz clock ticks parsed from a run comment.

    ``None`` stands for the start or the end of the run."""
    out = []
    for a, b in _VETO.findall(comment):
        out.append((None if not a.isdigit() else int(a), None if not b.isdigit() else int(b)))
    return out
