#!/usr/bin/env python3
"""Farm ClaudLand jobs out to the LSF batch system (``bsub``), one job per run.

Typical use (July 2002 ⁶⁰Co z-scan, first sub-run file of every run)::

    python scripts/farm.py extract 881-899 -o ~/cache/co60scan            # hit caches
    python scripts/farm.py reco 881-899 -o ~/cache/co60scan \\
           -- --calib ~/cache/calib.json --ml-pdf ~/cache/timepdf.npz --joint hit
    python scripts/farm.py reco 1518 --files 6 -o ~/cache/ge68 -- --energy-scale

* ``extract`` runs ``scripts/extract_hits.py``, ``reco`` runs ``scripts/reco_run.py``;
  everything after ``--`` is passed to that script unchanged.
* Runs are given as numbers or ranges (``881-899,1518``); the files are found
  through ``claudland.toml`` (``data_dir``, several directories allowed).
  ``--files N`` uses the first N sub-run files of each run (default 1).
* The raw files live on tape-backed HSM: reading them from many nodes at once
  thrashes the tape drive, so by default the files are recalled *sequentially*
  from the submit host (``cat > /dev/null``) before the jobs are submitted;
  ``--no-stage`` skips this when the files are known to be on disk.
* Output, logs and LSF stdout go to ``-o`` (must be on a filesystem the batch
  nodes mount -- on this cluster the home GPFS, not ``/cache``).  Job output
  files are ``<tool>_<run>.npz`` / ``.log``.
* ``--queue`` (default ``kamland``; ``bqueues`` lists them), ``-W`` run limit in
  minutes, ``-M`` memory limit in MB, ``--dry-run`` prints the ``bsub`` commands.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from claudland import config as _config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
TOOLS = {"extract": "scripts/extract_hits.py", "reco": "scripts/reco_run.py", "spallation": "scripts/spallation.py",
         "frames": "scripts/reco_frames.py"}
EXT = {"frames": ".parquet"}                 # output extension per tool (default .npz)
LSF_BIN = "/export/lsf/10.1/linux3.10-glibc2.17-x86_64/bin"


def parse_runs(spec: str):
    """``'881-899,1518'`` -> sorted list of run numbers."""
    runs = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            runs.update(range(int(a), int(b) + 1))
        else:
            runs.add(int(part))
    return sorted(runs)


def stage(files, log=print):
    """Recall *files* from tape one after the other (a no-op for files already on disk)."""
    for f in files:
        t = time.time()
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass
        log(f"  staged {os.path.basename(f)} in {time.time() - t:.0f} s")


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", choices=sorted(TOOLS))
    ap.add_argument("runs", help="run numbers / ranges, e.g. 881-899,1518")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="output directory, must be visible from the batch nodes (default: <cache_dir>/farm_<tool> of claudland.toml)")
    ap.add_argument("--files", type=int, default=1, help="number of sub-run files per run (default 1)")
    ap.add_argument("--file-range", default=None, metavar="A-B",
                    help="use sub-run files A..B (1-based, inclusive) of each run instead of the first --files")
    ap.add_argument("--tag", default="", help="suffix for the output names, e.g. b2 -> <tool>_<run>_b2.npz")
    ap.add_argument("--queue", default="kamland")
    ap.add_argument("-W", "--walltime", type=int, default=240, help="run limit in minutes")
    ap.add_argument("-M", "--memory", type=int, default=3000, help="memory limit in MB")
    ap.add_argument("-J", "--job-name", default=None, help="job name prefix (default: claudland_<tool>)")
    ap.add_argument("--no-stage", action="store_true", help="do not recall the files from tape first")
    ap.add_argument("--dry-run", action="store_true")
    ap.epilog = "Anything not recognised above (put it after --) is passed to the tool unchanged."
    args, tool_args = ap.parse_known_args()
    tool_args = [a for a in tool_args if a != "--"]

    cfg = _config.get()
    out = os.path.abspath(os.path.expanduser(args.out_dir or str(cfg.cache_dir / f"farm_{args.tool}")))
    os.makedirs(out, exist_ok=True)
    jobs = []
    for run in parse_runs(args.runs):
        allf = cfg.run_files(run)
        if args.file_range:
            a, b = (int(v) for v in args.file_range.split("-"))
            files = allf[a - 1:b]
        else:
            files = allf[:args.files]
        if not files:
            print(f"run {run}: no files found below {[str(d) for d in cfg.data_dirs]}", file=sys.stderr)
            continue
        jobs.append((run, [str(f) for f in files]))
    if not jobs:
        sys.exit("nothing to submit")
    print(f"{len(jobs)} runs, {sum(len(f) for _, f in jobs)} files")

    if not args.no_stage and not args.dry_run:
        print("recalling files from tape (sequentially) ...")
        stage([f for _, files in jobs for f in files])

    env = dict(os.environ)
    env["PATH"] = LSF_BIN + os.pathsep + env.get("PATH", "")
    prefix = args.job_name or f"claudland_{args.tool}"
    for run, files in jobs:
        name = f"{prefix}_{run}" + (f"_{args.tag}" if args.tag else "")
        stem = f"{args.tool}_{run}" + (f"_{args.tag}" if args.tag else "")
        ext = EXT.get(args.tool, ".npz")
        result = os.path.join(out, f"{stem}{ext}")
        log = os.path.join(out, f"{stem}.log")
        cmd = [PYTHON, os.path.join(REPO, TOOLS[args.tool]), *files, "-o", result, *tool_args]
        script = (f"cd {shlex.quote(REPO)}\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                  f"{' '.join(shlex.quote(c) for c in cmd)} > {shlex.quote(log)} 2>&1\n")
        bsub = ["bsub", "-q", args.queue, "-J", name, "-W", str(args.walltime), "-M", str(args.memory),
                "-o", os.path.join(out, f"lsf_{stem}.%J.out")]
        if args.dry_run:
            print(" ".join(bsub) + " <<EOF\n" + script + "EOF")
            continue
        r = subprocess.run(bsub, input=script, text=True, capture_output=True, env=env)
        print(f"run {run} ({len(files)} file{'s' if len(files) > 1 else ''}): {r.stdout.strip() or r.stderr.strip()}")


if __name__ == "__main__":
    main()
