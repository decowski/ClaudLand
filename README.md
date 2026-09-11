# ClaudLand — Python reconstruction framework for KamLAND raw data

A Python (numpy) toolkit, with one small optional C helper, that reads KamLAND *Serial Format* raw-data files
(`.sf` / `.sfz`), decompresses the ATWD waveforms, extracts hit times and charges,
fits the event vertex, estimates the visible energy and reconstructs muon tracks.  The name
is a nod to the Claude model that wrote it together with M.P. Decowski.  It was reverse-engineered
from the collaboration's C++ code (`SF/`, `WFComp/`, `Kat/`, `AKat/`) and the PhD
theses in `../Thesis/` (Detwiler 2005 for the reconstruction algorithms) and
validated on the March 2003 ⁶⁰Co z-scan runs in `../Run/co60-zscan/`.

```
ClaudLand/
├── claudland/                 the package
│   ├── sf.py                SF/SFZ file reader (events, banks, format strings)
│   ├── banks.py             typed Header / RunHeader / History / HitHeader decoders
│   ├── trigger.py           trigger-type bits
│   ├── wfcomp.py            ATWD waveform decompression (Huffman codec of .sfz)
│   ├── fastdecode.py        ctypes bridge to the C decoder (auto-compiled)
│   ├── muon.py              muon track fitter (first-light time + charge-pattern likelihood)
│   ├── config.py            locations of the KamLAND-private inputs (claudland.toml)
│   ├── runinfo.py           the run list (run-info.table)
│   ├── _cdecode.c           Huffman / raw ATWD decoder in C
│   ├── huffman_tables.py    static Huffman tables (generated from WFComp/trees.hh)
│   ├── pedestal.py          pedestal manager + per-waveform baseline
│   ├── tq.py                pulse finding: time and charge per waveform
│   ├── calib.py             sampling period (clock events), 1 p.e. charge, T0, gain ratios
│   ├── geometry.py          PMT positions / cable conventions (pmt_xyz.dat via claudland.toml)
│   ├── vertex.py            time-of-flight "window" vertex fitter (Kat/AKat style), two-medium ToF
│   ├── vertex_ml.py         maximum-likelihood vertex fitter with empirical time densities
│   ├── zscan.py             calibration from the z-scan: T0/Q0, light speeds, time PDFs
│   ├── energy.py            charge-based and hit-pattern visible energy
│   └── reco.py              EventReconstructor pipeline
├── scripts/
│   ├── sfdump.py            dump file structure / trigger census / waveforms
│   ├── reco_run.py          reconstruct a run (window or ML fitter), derive T0/Q0/energy scale
│   ├── extract_hits.py      decode a run once into a compact hit cache (.npz)
│   ├── zscan_calibrate.py   T0/Q0 from the centre run, light speeds and time PDFs from the scan
│   ├── zscan_evaluate.py    bias/resolution of the fitters versus source position
│   ├── plot_event.py        event display (waveforms, time spectrum, hit map) → PNG
│   └── plot_run.py          summary plots of a reconstructed run
├── cache/                   hit caches, calibration and evaluation outputs (not versioned)
├── tests/                   unit tests (+ integration tests on run 2279 if present)
├── claudland.toml             where the private inputs live (edit for your machine)
└── private/                 git-ignored: pmt_xyz.dat, run-info.table, huffman_tables.json, calibration JSONs
```

Requirements: Python ≥ 3.11 (``tomllib``), numpy; matplotlib for the plot scripts.  The
waveform decoder `claudland/_cdecode.c` (~200 lines, no dependencies) is compiled
automatically on first import with the system `cc` and loaded through
`ctypes`; without a compiler the pure-Python decoder is used instead (identical
output, ~20× slower decoding).  `python3 -m claudland.fastdecode` reports the
status and rebuilds; `CLAUDLAND_NO_CDECODE=1` forces the Python path.

## Quick start

```bash
cd Claude
python3 scripts/sfdump.py ../Run/co60-zscan/run_002279_000000_000001.sfz --summary
python3 scripts/sfdump.py ../Run/co60-zscan/run_002279_000000_000001.sfz --event 338 --header --waveforms

# reconstruct the run; derive per-channel time offsets from the known source position
# (RunHeader comment "+3.50 m"), set the energy scale on the 60Co peak, save calibration
python3 scripts/reco_run.py ../Run/co60-zscan/run_002279_000000_000001.sfz \
        --t0-source --energy-scale --v-ls 17.6 --save-calib calib_2279.json -o reco_2279.npz
python3 scripts/plot_run.py reco_2279.npz --source-z 350

# apply that calibration to another z position
python3 scripts/reco_run.py ../Run/co60-zscan/run_002278_000000_000001.sfz --calib calib_2279.json -n 1000
python3 scripts/plot_event.py ../Run/co60-zscan/run_002279_000000_000001.sfz --event 338 -o ev338.png

python3 tests/run_tests.py        # or: python3 -m pytest tests
```

Calibration from the full z-scan and the maximum-likelihood fitter (see below):

```bash
for r in $(seq 2270 2295); do                      # decode each run once (~100 s each, 5 in parallel is fine)
  python3 scripts/extract_hits.py ../Run/co60-zscan/run_00${r}_000000_000001.sfz -n 2500 -o cache/hits_$r.npz
done
python3 scripts/zscan_calibrate.py cache/hits_*.npz --center 2283 \
        --calib-out cache/calib_center.json --pdf-out cache/timepdf.npz
python3 scripts/zscan_evaluate.py cache/hits_*.npz --calib cache/calib_center.json --pdf cache/timepdf.npz -n 500
# full reconstruction of any run with the calibrated constants and the ML fitter
python3 scripts/reco_run.py ../Run/co60-zscan/run_002279_000000_000001.sfz \
        --calib cache/calib_center.json --ml-pdf cache/timepdf.npz -o reco_2279.npz
```

Python API:

```python
from claudland import SFReader, EventReconstructor

with SFReader("run_002279_000000_000001.sfz") as rd:
    for ev in rd:                       # SFEvent: run/subrun/event_number + dict of SFBank
        hdr = ev["Header"]              # raw bank; .unpack(), .array(dtype)
        break

rec = EventReconstructor("run_002279_000000_000001.sfz")
rec.prepare()                           # pedestals + clock periods from the run's calibration events
table = rec.run(max_events=2000)        # structured array, one row per physics event (EVENT_DTYPE)
r = rec[337]                            # RecoEvent of file index 337: r.hits, r.vertex, r.energy
```

## Configuration and private inputs (`claudland.toml`)

The package contains no collaboration data.  Everything KamLAND-specific is
looked up through one file, `claudland.toml`, found as `$CLAUDLAND_CONFIG`, then
`claudland.toml` in the working directory or a parent, then next to the package,
then `~/.config/claudland/claudland.toml` (`claudland.config`):

| key | default | what |
|---|---|---|
| `pmt_table` | `private/pmt_xyz.dat` | PMT coordinates, `cable x y z` in cm (Kat `pmt_xyz.cc`) |
| `run_info` | `private/run-info.table` | the run list (`claudland.runinfo` parses it) |
| `huffman_tables` | `private/huffman_tables.json` | code tables of the `.sfz` waveform compression (from `WFComp/trees.hh`) |
| `data_dir` | `../Run` | raw `.sf`/`.sfz` files, searched recursively (`Config.find_run(2283)`) |
| `cache_dir` | `cache` | derived hit caches, fit tables, plots |
| `tq`, `time_pdf` | `cache/calib_center.json`, `cache/timepdf.npz` | default calibration and time PDFs; the scripts use them when present |

Relative paths are resolved from the directory holding the file.  Importing
the package works without any of these files; using one that is missing raises
`PrivateInputMissing` naming the file and the configured location.  `private/`
and `cache/` are git-ignored, so a clone of the repository needs the three
private files copied in and `data_dir` pointed at the data.

Derived quantities in this README and the plots in `examples/` come from
KamLAND data (calibration runs 2270–2295 of March 2003 and physics run 1467 of
October 2002); check with the collaboration before making them public.

## Detector geometry conventions

`pmt_xyz.dat` (private, located through `claudland.toml`) is the Kat constants table (identical to
`Kat/src/pmt_xyz.cc`): 2126 cables, x y z in cm, inner tubes on the 850 cm
stainless-steel sphere they are bolted to.  The photocathode / first dynode,
where the photo-electron is produced and where the light path effectively
ends, is ~20 cm further in, which is the origin of the 830/850 rescaling in
Kat's `KPmtTable`; `PMTTable.load(radius_scale=830/850)` applies the same
shift (inner tubes only).  All inner tubes are 20-inch Hamamatsu envelopes:
the "17-inch" tubes (cables 0–1324) are the fast-timing version with the
photocathode masked at the edges to a 17-inch equivalent area, the "20-inch"
tubes (1325–1878) use the full photocathode.  The two photocathode areas
(radii 21.83 and 23.0 cm) enter the solid-angle factor of the energy estimator.

Cable layout of the table, cross-checked against the tube types in
`AKat/util/serial.table` (column `pmt_type`, PMT serial prefixes `17-`, `20-`,
`OD-`, `5-1`):

| cables | n | type | where |
|---|---|---|---|
| 0–1324 | 1325 | 17-inch (masked photocathode, fast) | inner sphere, r = 850 cm |
| 1325–1878 | 554 | 20-inch (full photocathode) | inner sphere, r = 850 cm, interleaved in θ/φ with the 17-inch tubes |
| 1879–2119 | 241 | outer detector 20-inch | water Čerenkov veto: bottom plane z = −930 cm (rings ρ = 200–800 cm), side wall ρ = 890 cm at z = ±350, ±550, ±750 cm, top plane z = +930 cm, plus 16 tubes at z = +940 cm, ρ = 120–180 cm around the chimney |
| 2120–2125 | 6 | 5-inch neck tubes | chimney, z = +1143 cm, ρ = 49 cm |
| 2126–2131 | 6 | CCD cameras (no PMT) | not in `pmt_xyz.dat` |

Only cables 0–1878 enter the inner-detector reconstruction; the neck tubes are
outside the sphere and are not part of it.

## The data format

An SF file is a concatenation of *events*; each event is an `SFdata` record whose
payload is a sequence of named `SFdata` records (*banks*).  Every record is
`[vsize total][vsize name_len][name][vsize form_len][form][data][vsize_r total]`,
where `vsize` is a 1/3/5/7-byte variable-length integer (`0xFF` escapes) and
`form` a nibble-packed field-format string (`2CIB4ILI8BIL` = 2 chars, int, short,
4 ints, long, int, 8 shorts, int, long).  The first nibble of the format carries
the byte order: the `Header` bank is **big-endian**, everything else little-endian.
The event name is the 10-byte big-endian triple (run int32, subrun int16, event int32).

Banks found in the ⁶⁰Co files (`claudland.banks` decodes the first four):

| bank | content |
|---|---|
| `RunHeader` | first event only: versions, run, start time, shifters, run type (`source-60Co`), comment (`+3.50 m` = source z) |
| `Header` | run/subrun/event, unix time (+µs, ns), 40 MHz timestamp, trigger type, Nsum values, error status, time since previous trigger |
| `History` | trigger records (type, Nsum, OD Nsums) within the event |
| `HitHeader` / `AntiHitHeader` | ID / OD hit list: `u16` per hit, `cable = w & 0xfff`, `n_waveforms = (w>>12) & 7` |
| `ConnectionTable` | first event: cable → (crate, slot, channel) of the front-end electronics |
| `CmpPedestal` | first event: 19200 × 128 `u16` pedestals used by the compressor |
| `CmpATWD` / `CmpAntiATWD` | compressed waveforms in `HitHeader` order (replace `ATWD`/`AntiATWD` of `.sf` files) |

Trigger types (`claudland.trigger`): each run starts with 50 × `PedestalA/B`, 50 ×
`ClockA/B`, 50 × `ForcedAcqA/B` events (all ~1810 channels read out, three gains);
the source events carry `IDtoOD|Prescale` (0x8000100, with or without `IDHistory`);
93% of the records are hit-less `IDHistory` trigger records.

### Waveform compression

Each compressed block is `u8 size, u8 launch_offset (bit7 = sign), u8 flags
(gain = bits 0-1, ATWD A/B = bit 2), u8 info, u16 nbits, code...`.  `nbits` is
padded to a multiple of 32; `nbits == 1376` marks an uncompressed block (43
little-endian `u32` words, 3 × 10-bit samples per word).  Otherwise the 128
samples are decoded with the static Huffman code of Dwyer/Batygov
(`DiffDetEntStatHuffComp0.1`, MSB-first bits in little-endian 32-bit words), the
inverse "difference + reflect + power-compression" transform is applied and the
compression pedestal of the channel is added back (mod 1024).  A round-trip test
against a Python port of the encoder is in `tests/test_wfcomp.py` (the tables themselves are read from the private JSON, see above); on data the
decoded pedestal-trigger waveforms sit within a few ADC counts of the pedestal table.

## Reconstruction chain (`claudland.reco.EventReconstructor`)

1. **Constants** (`prepare`): compression constants from the first event; average
   pedestal per (cable, ATWD, gain) from the pedestal-trigger events with the
   quality cuts of `KPedestalFifoManager` (all 3758 ID channel×ATWD combinations
   are measured in run 2279); ATWD sampling period per (cable, ATWD) from the FFT
   of the 40 MHz clock waveforms (16.66 samples per 25 ns → 1.50 ns/sample,
   Detwiler §4.2.3); dead channels from the hit occupancy of the first physics events.
2. **TQ** (`tq.fast_tq`, vectorised over all waveforms of an event): pedestal and
   baseline subtraction, 5-point running-average smoothing (Inoue's `KMultiTQ`),
   pulses = positive regions of the smoothed waveform with height ≥ 5 counts and
   area ≥ 15 % of the total (Detwiler §4.2.2); charge = summed area; time = 50 %
   constant-fraction crossing of the first pulse (`t_lead`/`t_peak` also kept).
   `tq.multi_tq` is a line-by-line port of `KMultiTQ::get_multi_tq` for reference.
   Conversions: `t[ns] = sample × bin_ns − launch_offset × 25 ns − T0`,
   `q[p.e.] = ADC_sum / 200 × gain_factor` (the 1 p.e. peak of the high-gain
   charge spectrum lands at 1.0 with 200 ADC·sample; `--q0` fits it per channel).
3. **Vertex** (`vertex.VertexFitter`): weighted Gauss-Newton fit of (x, y, z, T) to
   `t_i = T + |P_i − r| / v_eff` using only hits within a time window around the
   peak of the ToF-corrected time distribution; the window shrinks 40 → 12 ns.
   Start point: charge-weighted centroid scaled by 1/0.62 (Detwiler §4.3.1).
   ~2 ms per event.
4. **Energy** (`energy.EnergyEstimator`): expected light per tube
   `η_i f_i(r)` with `f_i ∝ A_i (0.1 + 0.9 cos θ_i) e^{−d_i/Λ} / d_i²` (Detwiler
   Eq. 4.8, Λ = 25 m, η = 0.220 / 0.293 p.e./MeV for 17"/20" tubes).
   `e_charge` = dark-subtracted charge in a (−15, +85) ns window / Σ η_i f_i(r);
   `e_hit` = maximum-likelihood energy from the hit/no-hit pattern (Eq. 4.6).
   `--energy-scale` normalises both to 2.506 MeV on the source peak.
5. **Calibration from data**: `calibrate_t0(hits)` (per-channel median ToF
   residual, either self-consistent or with the vertex fixed at the known source
   position), `calibrate_q1pe(hits)`; constants are saved/loaded as JSON
   (`calib.TQCalibration`).

Throughput on this machine: ~35 ms per source event (≈800 waveforms) including
decompression, i.e. a full 268 MB run (≈5100 physics events) in about 3 minutes,
plus 20 s for the calibration block.

## Validation on the ⁶⁰Co z-scan

Run 2279 (source at (0, 0, +350 cm)), source-like events (nhit > 600) from the
first 1500 physics events, effective light speed 16.95 cm/ns (Kat default):

| step | z (median) | robust σ_z | x, y | σ_t |
|---|---|---|---|---|
| default constants | 324 cm | 33 cm | −35, +45 cm | 6.3 ns |
| + T0 from fitted vertices | 316 cm | 20 cm | −25, +45 cm | 5.8 ns |
| + T0 with vertex fixed at the source | 336 cm | 18 cm | −2, +10 cm | 5.7 ns |

Scanning the light speed on run 2279 gives z = 317 / 336 / 357 / 387 cm for
16.0 / 16.95 / 18.0 / 19.5 cm/ns; the framework default is therefore
`--v-ls 17.6`.  Full run 2279 with `--t0-source --energy-scale --v-ls 17.6`
(5104 physics events, 5093 fitted, 3 min): z = 339.6 cm, robust σ_z = 19 cm,
ρ = 23 cm; energy resolution at the 2.506 MeV peak 5.2 % (charge) / 4.6 %
(hit pattern).  The same calibration file applied to the other positions of the
scan (1000 events each, `examples/reco_2279.png` shows the run summary):

| run | source z | reconstructed z | bias | robust σ_z | E_charge / E_hit |
|---|---|---|---|---|---|
| 2280 | 300 cm | 293.9 cm | −6 cm | 19.7 cm | 3.68 / 2.17 MeV (unscaled) |
| 2279 | 350 cm | 339.6 cm | −10 cm | 19.1 cm | 2.51 / 2.51 MeV (scale run) |
| 2278 | 400 cm | 387.4 cm | −13 cm | 21.3 cm | 3.67 / 2.16 MeV (unscaled) |
| 2276 | 450 cm | 433.6 cm | −16 cm | 18.8 cm | 3.62 / 2.14 MeV (unscaled) |
| 2274 | 500 cm | 483.8 cm | −16 cm | 21.0 cm | 2.45 / 2.46 MeV (scaled) |

The remaining 2–3 % inward bias grows with z and is the kind of effect Detwiler
tuned away with separate scintillator / buffer-oil light speeds
(`VertexFitter(v_bo=...)` enables the two-medium model); the unscaled energies
are stable to 2 % between z = 300 and 500 cm, so the position correction of the
light-collection model works at that level.

## Calibration from the centre run and the maximum-likelihood fitter

Policy: **all constants come from the centre run 2283** (source at the origin);
the other 25 runs 2270–2295 (z = +6.0 m … −5.75 m) are used only as a check.

1. **Hit caches** (`extract_hits.py`): each run is decoded once into a compact
   per-hit table (samples, ADC sums, launch offsets, the run's own ATWD
   sampling periods) so that every later step runs in seconds.
2. **T0 and Q0** (`zscan.calibrate_center`): with the source at the origin every
   tube is at the same distance, so the per-channel time offset is the *mode*
   of the hit-time residual with the vertex fixed at (0, 0, 0), independent of
   the light speed (3734 channel×ATWD offsets, rms 10.5 ns, range −21 … +31 ns).
   Q0 is the mode of each channel's high-gain ADC spectrum (single-p.e. peak;
   median 238 ADC·sample for 17-inch, 257 for 20-inch tubes).
3. **Light yield per tube** (`zscan.calibrate_light_yield`): from the hit
   probability p_i of each tube in the centre events, μ_i = −ln(1 − p_i),
   η_i = (μ_i − dark_i)/2.506 MeV (Detwiler Eq. 4.7–4.9 at one position); a
   charge-based yield η_q,i (mean p.e. per MeV) is stored for the charge
   estimator, and dark_i from the pre-signal window.  Means: 0.19 (17") and
   0.24 (20") p.e./MeV, 1867 live tubes.
4. **Time densities** (`zscan.build_time_pdf --center-only`): the residual
   distribution t − T − tof from the centre run in 0.5 ns bins, 2 tube types ×
   4 charge bins.  The 1 p.e. density of a 17-inch tube has a 13.5 ns FWHM —
   scintillator emission, not electronics — and a long tail; the first of
   several photo-electrons arrives earlier and sharper (10.5 ns FWHM for 3–6
   p.e.).  The constant-fraction time gives the sharpest density of the three
   timing definitions in the cache.
5. **Light speed**: the one quantity that *cannot* be measured at z = 0.  It is
   an external input, 19.4 cm/ns (≈ c/1.55), the value obtained once from the
   peak of the residual versus path length over the scan
   (`zscan.fit_velocities`, still available with `--max-z`).  A two-medium
   ToF (scintillator inside the 6.5 m balloon, buffer oil outside) is
   implemented; the scan constrains the oil speed only weakly (19.0 cm/ns).
6. **Likelihood fitter** (`vertex_ml.MLVertexFitter`): maximises
   Σᵢ log φₖ(tᵢ − T − tofᵢ(r)) over (x, y, z, T) with the tabulated densities
   (the "V2" idea of Batygov's thesis, App. A), a damped Newton iteration with
   the analytic derivative of log φ and a numerical ToF Jacobian, started from
   the window fitter.  Every hit contributes with its proper asymmetric
   weight, so no hard time window and no hand-tuned light speed are needed;
   the inverse Hessian gives a per-event position uncertainty.  ≈ 10 ms/event.

7. **Joint charge + time likelihood** (`vertex_ml.ChargeTimeVertexFitter`):
   adds to the time term the likelihood of the hit pattern, with
   μ_i = E η_i f_i(r) + dark_i from the light-collection model and the per-tube
   yields of step 3 — either Bernoulli on hit / no hit (Detwiler Eq. 4.6,
   `charge_model="hit"`) or a continuous Poisson on the measured charge
   (`"poisson"`).  The parameters are (x, y, z, T, ln E), so the visible energy
   comes out of the same fit (≈ 7 ms/event).

### Check on the other 25 positions (500 source events per run, `examples/zscan_eval.png`)

A = window fitter with a single effective speed (18.5 cm/ns, the value that
minimises the scan bias — so A is *not* a centre-only calibration),
C = time likelihood, D = joint time + hit-pattern likelihood,
E = joint time + Poisson-charge likelihood.

| |z| ≤ 550 cm | mean \|z bias\| | rms z bias | mean σ_z | mean σ_x | median ρ (source on axis) |
|---|---|---|---|---|---|
| A window, tuned single v | 3.6 cm | 4.3 cm | 18.9 cm | 17.6 cm | 20.8 cm |
| C likelihood, time only | 3.1 cm | 3.5 cm | 17.0 cm | 15.7 cm | 18.4 cm |
| D likelihood, time + hit pattern | **2.0 cm** | **2.3 cm** | **15.8 cm** | **15.4 cm** | **18.0 cm** |
| E likelihood, time + Poisson charge | 2.0 cm | 2.4 cm | 16.7 cm | 16.9 cm | 19.9 cm |
| C with densities from the whole scan (distance-binned) | 2.0 cm | 2.3 cm | 17.2 cm | 15.3 cm | 18.1 cm |

The hit pattern carries genuine position information: adding it removes the
slow outward drift of the time-only fit (+5 cm at +5 m, −5 cm at −5 m) and
improves the z resolution by 7 %, without any information from the scan.  The
Poisson charge term is weaker (PMT charge resolution and multi-p.e. pile-up are
not modelled), so the Bernoulli hit pattern is the default joint model
(`reco_run.py --ml-pdf ... --joint hit`).  The joint fit's energy
(2.53 ± 0.02 MeV over |z| ≤ 550 cm, 0.9 % rms, 5.3 % resolution) agrees with
the charge estimator at every position.

With centre-only constants the joint fitter stays within 5 cm of the true
position from −575 to +600 cm (Detwiler's Fig. 4.4 quotes < 5 cm within
5.5 m around a 2 cm offset).
Comparing A and B shows why the old window fitters needed a slow "effective"
light speed: it is a tail compensation, not a property of the light.  The
15–17 cm resolution is close to the intrinsic size of a ⁶⁰Co event (two
1.2 MeV gammas with a ~20 cm mean free path).  Moving the inner PMTs to the
830 cm photocathode radius (`radius_scale=830/850`) changes the likelihood
result by +1 cm rms bias and the window fitter by −0.5 cm; the default stays
at the 850 cm table because the T0 calibration absorbs the common radial
offset.

## Mechanical tolerances: what the source data can and cannot tell

`scripts/zscan_survey.py` fits every inner tube's position offset dP_i and time
offset dT_i from the run-by-run modes of its time residual (vertex fixed at the
source): m_ik = dT_i + u_ik·dP_i / v.  Results (`examples/survey.png`):

* **Per tube** the survey precision is 23 cm radially and 45 cm tangentially
  (17-inch tubes; worse for 20-inch), because along an on-axis scan the
  direction from the source to a tube changes by at most ~40° and hardly at all
  for tubes near the poles.  The observed scatter of the fitted offsets (13 cm
  radial rms) is entirely consistent with that uncertainty, i.e. no per-tube
  deviation is resolved.  Bolt-pattern and sphere tolerances at the few-cm level
  therefore have no visible effect: a 10 cm radial misplacement is 0.5 ns and is
  absorbed into that tube's T0 (whose channel-to-channel spread is 10 ns) with a
  residual position-dependent error well below the 17 cm event resolution.
* **Averaged over rings in θ** the fit does show a coherent pattern, −8 cm at
  the equator and 0 at the poles with 1 cm errors, but it is degenerate with
  the light speed: with v = 19.0 cm/ns the equatorial value halves (−4 cm) and
  with 19.8 cm/ns it grows to −13 cm.  For a source moving along the axis, an
  oblate sphere and a change of the light speed produce the same even-in-z
  residual pattern.  Off-axis deployments (the later 4π calibration system) would
  break this degeneracy; with the axial scan alone the sphere shape is not
  separable from the effective light speed.

## Muon track reconstruction (`claudland.muon`)

Through-going muons deposit ~630 p.e. per cm of scintillator on the 17-inch
tubes (2.5 GeV for a central track) and saturate every high-gain and most
medium-gain waveforms.  `muon_hits` therefore collapses the hit list to one
entry per tube: the **time** from the earliest high-gain pulse (as in the
theses: highest gain regardless of saturation, T0 applied) and the **charge**
from the highest gain that is *not* saturated (the low-gain channel never is;
both file formats store all three gains for muon events).  The selection is
the standard one, `Q17 ≥ 10 000 p.e.` or (`Q17 ≥ 500 p.e.` and `N200_OD ≥ 5`),
with `N200_OD` the maximum number of outer-detector hits in a 200 ns window
(the total OD multiplicity lets ⁶⁰Co source events with accidental OD hits
through).

`MuonTrackFitter` fits a straight line through the PMT sphere (r = 850 cm):
two points on the sphere (2 × 2 tangent-plane coordinates) and the entrance
time, i.e. the model of Ichimura/Abe/Detwiler/O'Donnell and of
`Kat/src/KatMuonFitter.cc` written as one explicit optimisation problem:

* **Time model** – the *earliest* light that can reach tube *i* comes from the
  point of the track that sees it under the Cherenkov angle `cos θc = 1/n`:
  `t_i = t0 + (z_i + ρ_i tan θc)/c`, or from the entrance/exit point when that
  emission point lies outside the track.  A single effective index n = 1.5 is
  used (scanning 1.40–1.60 on run 1467 changes σ_t by < 0.1 ns; the
  impact-parameter table of Kat, 1.65 → 1.4, fits worse and lets the fit trade
  impact parameter against light speed, which piles tracks up at the balloon
  radius; a free per-event *n* is degenerate with t0 for central tracks).
* **Charge model** (`MuonChargeModel`) – expected charge per 17-inch tube for a
  minimum-ionising track: 629.4 p.e./cm of scintillation along the LS chord,
  collected with the point-source acceptance of the energy estimator, plus
  31.45 p.e./cm of Cherenkov light along the whole track inside r = 830 cm,
  half on the cone (∝ 1/d from the emission point) and half isotropic
  (scattered/reflected light, which dominates the diffuse blob of buffer-oil
  clippers).  Measured and expected charges are compared in log space after
  profiling a global scale, so showers change the normalisation but not the
  pattern.
* **Global search** – all pairs of 150 Fibonacci grid points on the sphere
  (22 000 oriented tracks) are scored by the number of hits within 12 ns of the
  first-light prediction (t0 profiled out) minus a penalty for the total
  17-inch charge against the minimum-ionising expectation for that geometry;
  the best 20 distinct pairs are re-ranked with the charge-pattern score and
  the best 3 refined on finer grids.
* **Outer detector** – the OD tubes are not part of the track fit: their light
  is reflected and diffuse (it arrives ~60 ns after the ID first light with a
  70 ns spread), so they cannot sharpen the entry or exit point.  Their *time
  ordering* does follow the muon, though: on the well-measured LS tracks the OD
  hit time grows with the along-track coordinate of the tube with slope
  0.030 ns/cm (1/c = 0.033) and the entrance-side tubes fire first in 62 of 64
  tracks.  Each candidate therefore carries an OD term, the capped mean squared
  residual of `t_OD − t0 − z/c` about its median (σ = 25 ns, weight 0.1), which
  fixes the orientation of short oil clippers where the ID times barely
  distinguish the two ends.  `--no-od` switches it off.
* **Chimney** – the six 5-inch tubes at the top of the chimney (cables
  2120–2125) see the scintillator in the chimney.  As in
  `KatMuonFitter::checkChimney`, a muon with `Q_5inch ≥ 100 p.e.` is flagged
  (`chimney`), gets an extra seed with the entrance at the charge centroid of the
  ID hits above z = 800 cm, and a score penalty `0.1 × ((d − 200 cm)/200 cm)²`
  when the track passes more than 2 m from the top of the sphere; the closest
  approach `chimney_dist` is stored for every track.  In our 143 muons two are
  flagged (1671 and 1234 p.e.): a vertical LS muon entering 3.3 m off axis and a
  horizontal track 70 cm below the chimney — the penalty does not move the first,
  whose times and charges prefer the off-axis entrance, so light reaching the
  5-inch tubes does not by itself imply a passage through the chimney.
* **Joint robust fit** – damped Gauss-Newton on Σ w_t r_t² + λ Σ w_q r_q² with
  Welsch time weights `exp(−r²/2σ²)` (σ = 30 → 4 ns) and Huber charge weights;
  numerical Jacobians; the flipped orientation is always tried as well.
  Candidates are ranked by `used-hit fraction − total-charge term − 0.3 ×
  pattern score − 0.1 × OD term`.

Why the charge is needed: on the times alone a long track through the
scintillator (n ≈ 1.6) and a short track hugging the PMT sphere (n ≈ 1.4) are
nearly degenerate — a 760 000 p.e. event fitted equally well as a 500 cm oil
clipper — and for oil clippers the orientation (which end is the entrance) is
decided by a handful of tubes.  The total charge separates the first pair by a
factor 50, the per-tube pattern (forward-peaked light at the exit) fixes the
orientation, and a class of scintillator-edge tracks (100–400 k p.e., b ≈
550–620 cm) is only found with the pattern term.

Outputs per muon (`MUON_DTYPE`): entrance/exit on r = 850, direction, zenith
angle, impact parameter, track lengths in LS (chord through r = 650) and buffer
oil (chord through 830 minus LS, as in Kat), Q17/Q20/Q_OD, σ_t and used-hit
fraction, the charge ratio to the minimum-ionising expectation, the residual
charge `ΔQ = Q17 − 629.4 L_LS − 31.45 L_BO` (showering muons: ΔQ > 10⁶ p.e.),
the OD timing term, the chimney flag and the distance to the chimney.

Results on run 1467 (157 s, 43 muons, `examples/muon_ev567_run1467.png`,
`examples/muon_clipper_ev2422_run1467.png`, `examples/muons_run1467.png`):

| quantity | value |
|---|---|
| tracks converged / through the LS | 41 / 24 |
| σ_t of the used hits (median) | 3.4 ns, ~70 % of the 17-inch tubes within ±8 ns |
| (Q17 − 31.45 L_BO) / L_LS, median over LS tracks | 646 p.e./cm (theses: 629.4, independent charge scale) |
| Q17 / L_BO, buffer-oil tracks | 48 p.e./cm (theses: 31.45) |
| up-going tracks | 2 of 41, both near-horizontal (cos zenith −0.04 and −0.23); 14 of 43 with the times alone |
| rate | 43 / 157 s = 0.27 Hz (published: 0.34 Hz total, 0.20 Hz through the LS) |
| time per muon | ~3 s (global scan 1 s, joint fit 2 s) |

On the 26 ⁶⁰Co z-scan runs of 2003 (7160 s, 17- and 20-inch tubes; the source
runs prescale all triggers, so the recorded muon rate of 0.014 Hz is not
physical): with the standard selection 138 of 236 candidates are source events
with 5–7 accidental OD hits inside 200 ns, hence the `--od-trigger` option
(an OD trigger bit for the low-charge branch), which leaves 100 muons
(`examples/muons_zscan2003.png`): 96 converge, 52 through the LS with a median
(Q17 − 31.45 L_BO)/L_LS of 629 p.e./cm, 39 oil tracks at 47 p.e./cm, σ_t 3.6 ns,
7 up-going tracks of which 3 below cos zenith = −0.25 (12 without the OD timing
term).  The 17-inch light yield is thus the same within 3 % in October 2002
(17-inch only) and March 2003.

Not done: per-run yield calibration (`KatMuonTrackChargeTable`), the Kat
"badness" flags, showering-muon handling beyond ΔQ, and any
absolute angular-resolution estimate (the theses point out there is no
calibration source for it; spallation products lie within 3 m of the track for
~95 % of the events in their analyses).

```bash
python3 scripts/muon_fit.py ../Run/run_001467_000000_000001.sf --calib cache/calib_center.json \
        -o cache/muons_1467.npz --keep-hits          # --time-only / --index kat / --seed cluster to compare
python3 scripts/plot_muon.py cache/muons_1467.npz --event 566 -o ev566_mu.png
python3 scripts/plot_muon.py cache/muons_22*.npz --od-trigger -o muons_2003.png   # source runs
```

## Speed

Single core (Apple silicon), run 2279 with ~760 hits per source event, default
chain (`--joint hit`: window prefit → ML time fit → joint charge+time fit):

| stage | pure Python | with C decoder |
|---|---|---|
| start-up (`prepare`: 100 pedestal + 100 clock events ≈ 2.2 M waveforms) | 23 s | 2.2 s |
| per physics event | 48 ms (21 events/s) | 15 ms (65 events/s) |
| waveform decoding per event | 30 ms | 1.3 ms |
| pedestal, baseline, TQ per event | 7 ms | ~4 ms |
| vertex fits per event (prefit + ML + joint) | ~10 ms | ~10 ms |

The decoder is bit-exact in both paths (`tests/test_cdecode.py` compares them
on both file types).  What remains is the numpy-vectorised fitting, which is
call-overhead bound at ~1 ms per Newton iteration; the remaining large gain
is running files in parallel processes, since events are independent.  Numba
was tried for the decoder (same 20×) and dropped in favour of the C helper to
avoid the dependency, which the Homebrew Python cannot install without a venv.

## Uncompressed `.sf` files (physics run 1467, October 2002)

`SFReader`/`WaveformDecompressor` handle the pre-compression format
transparently: the `ATWD`/`AntiATWD` banks hold 43 little-endian 32-bit words
per waveform (header word with A/B, gain, launch offset and samples 126–127;
then 42 words with three 10-bit samples each), decoded by
`wfcomp.decode_atwd_bank`.  There is no `ConnectionTable`/`CmpPedestal`, so the
pedestals come from the run's 100 pedestal-trigger events alone.  Run 1467
(24 h, grade 0; the first 268 MB sub-file covers 157 s) has 1312 inner cables
— the 17-inch tubes only, the 20-inch tubes were switched on at run 2194 —
and its physics triggers are `IDtoOD|Prompt` (0x0a000000) and
`IDtoOD|Delayed` (0x09000000).  `reco_run.py ... --calib cache/calib_center.json
--ml-pdf cache/timepdf.npz --joint hit` reconstructs its 2932 physics events in
36 s — here the vertex fits dominate, the raw banks were cheap to decode already
(`examples/reco_1467.png`); the z–ρ² map shows the balloon surface, the
chimney and the bottom of the balloon as in Detwiler's Fig. 4.5.



* The default (window) fitter has no time-vs-charge or multi-photon corrections;
  the likelihood fitter covers these through its charge-binned densities.  The
  densities are averaged over the source runs (γ events of 2.5 MeV); a
  dependence on event energy/particle type is not modelled.
* Per-channel Q0 is only available after `zscan_calibrate.py` (default is a
  global 200 ADC·sample).
* The hit-pattern energy assumes global η per tube type and a uniform dark rate;
  a proper η_i fit needs the full z-scan (Detwiler Eq. 4.9).
* Bad-channel handling is occupancy-based only; run-dependent bad-channel tables
  from the collaboration database are not used.
* Only the first sub-run file of each run is handled; `SFReader` reads one file
  (chain the files yourself if needed).
* MoGURA-era banks and the BWT-compressed format strings of the `SF` library are not
  implemented (never used in these files).

## License

BSD 3-Clause, see `LICENSE`; `CITATION.cff` gives the citation for the software.
