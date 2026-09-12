"""Per-event pandas frames of reconstructed runs (written by ``scripts/reco_frames.py``).

One row per physics event, all sub-run files of a run concatenated.  Columns:

``run, file, index, event, unix_time, timestamp, trigger, nsum, nsum_max``
    bookkeeping: sub-run file number, sequential index in that file, event number,
    Unix time (s), 40 MHz time stamp, trigger word, N_sum values.
``nhit, nhit_od, nsat, nwave, q17, q20, q_od, q_total``
    multiplicities and charges (p.e.) of the event (all gains, one entry per tube).
``is_muon, retrigger, od_only, n200_od``
    standard muon selection (Q17 >= 10 000 p.e., or Q17 >= 500 p.e. with N200_OD >= 5);
    ``retrigger``: muon-like with Q17 < 10 000 p.e. within 100 us of the previous muon;
    ``od_only``: Q17 < 10 000 p.e. and fewer than 500 ID hits (an outer-detector muon, no track fit).
``dt_prev_us, dt_muon_us, prev_muon, dist_track``
    time since the previous physics event, time since the previous real muon, the
    ``event`` number of that muon and the distance of this event's vertex to its track (cm).
``x, y, z, t0, r, sigma_t, n_used, vertex_ok, e_hit, e_charge, q_window, n_window``
    joint time + hit-pattern likelihood vertex and the two visible-energy estimates (MeV,
    scale of the calibration, see ``energy_unit`` of the calibration meta data).
``mu_*``
    muon track of muon events (NaN otherwise): entrance ``mu_ex/ey/ez`` and exit ``mu_xx/xy/xz``
    on the PMT sphere, direction ``mu_ux/uy/uz``, ``mu_cos_zenith``, ``mu_impact``, path lengths
    ``mu_l_ls`` / ``mu_l_bo`` (cm), ``mu_t0``, ``mu_sigma_t``, ``mu_frac_used``, ``mu_converged``,
    ``mu_delta_q`` (charge above minimum-ionising), ``mu_charge_scale``, ``mu_chimney``.

Example::

    from claudland import frames
    df = frames.load(1550)                                  # every frame of run 1550 in cache_dir/frames
    mu = df[df.is_muon & ~df.retrigger & df.mu_converged & (df.mu_l_ls > 0)]
    n  = df[~df.is_muon & df.vertex_ok & df.e_hit.between(1.8, 2.6) & (df.dt_muon_us < 2000)]
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional, Union

import pandas as pd

__all__ = ["load", "frame_dir", "frame_files"]


def frame_dir(run: Optional[int] = None) -> str:
    """``<cache_dir>/frames[/run00NNNN]``."""
    from . import config
    d = os.path.join(str(config.get().cache_dir), "frames")
    return os.path.join(d, f"run{run:06d}") if run is not None else d


def frame_files(run_or_pattern: Union[int, str, Iterable[str]]) -> List[str]:
    """Parquet files of a run number, a glob pattern, or an explicit list."""
    if isinstance(run_or_pattern, int):
        return sorted(glob.glob(os.path.join(frame_dir(run_or_pattern), "*.parquet")))
    if isinstance(run_or_pattern, str):
        return sorted(glob.glob(os.path.expanduser(run_or_pattern)))
    return sorted(set(sum([glob.glob(os.path.expanduser(p)) for p in run_or_pattern], [])))


def load(run_or_pattern: Union[int, str, Iterable[str]], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Concatenate the frames of a run (or matching a pattern) into one DataFrame sorted by time stamp."""
    files = frame_files(run_or_pattern)
    if not files:
        raise FileNotFoundError(f"no frames for {run_or_pattern!r}")
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    if "run" in df and "timestamp" in df:
        df = df.sort_values(["run", "timestamp"], kind="stable").reset_index(drop=True)
    return df
