"""The C decoder must reproduce the pure-Python decoder bit for bit."""
import os
import numpy as np
import pytest

from claudland import SFReader, fastdecode
from claudland.wfcomp import WaveformDecompressor, WaveformBatch
from claudland.pedestal import PedestalManager

from claudland import config as _config

_RAW = _config.get().find_run(1467)
RAW_FILE = str(_RAW) if _RAW else "run_001467_missing.sf"


def _events(path, n, need=("HitHeader",)):
    rd = SFReader(path)
    it = iter(rd)
    first = next(it)
    evs = []
    for ev in it:
        if any(k in ev for k in need):
            evs.append(ev)
        if len(evs) >= n:
            break
    return first, evs


def _compare(path, n_events):
    first, evs = _events(path, n_events, need=("HitHeader", "AntiHitHeader"))
    dec_c = WaveformDecompressor(use_c=True)
    dec_py = WaveformDecompressor(use_c=False)
    dec_c.load_constants(first); dec_py.load_constants(first)
    n = 0
    for ev in evs:
        for det in ("ID", "OD"):
            b = dec_c.decompress_arrays(ev, det)
            ref = WaveformBatch.from_list(dec_py.decompress(ev, det))
            assert len(b) == len(ref)
            assert np.array_equal(b.cable, ref.cable) and np.array_equal(b.atwd, ref.atwd)
            assert np.array_equal(b.gain, ref.gain) and np.array_equal(b.launch, ref.launch)
            assert np.array_equal(b.samples, ref.samples)
            n += len(b)
    assert dec_c.n_bit_mismatch == dec_py.n_bit_mismatch
    assert dec_c.n_raw == dec_py.n_raw and dec_c.n_compressed == dec_py.n_compressed
    return n


def test_c_decoder_available():
    assert fastdecode.available, fastdecode.build_error


def test_compressed_file_matches_python(run_file):
    assert _compare(run_file, 6) > 1000


def test_raw_file_matches_python():
    if not os.path.exists(RAW_FILE):
        pytest.skip("uncompressed run 1467 not present")
    assert _compare(RAW_FILE, 6) > 1000


def test_gain_selection(run_file):
    first, evs = _events(run_file, 3)
    dec = WaveformDecompressor(use_c=True)
    dec.load_constants(first)
    for ev in evs:
        b = dec.decompress_arrays(ev, "ID", gains=(0,))
        assert len(b) == 0 or set(np.unique(b.gain)) == {0}
        ball = dec.decompress_arrays(ev, "ID")
        assert len(b) == int((ball.gain == 0).sum())


def test_pedestal_batch_matches_scalar(run_file):
    from claudland.banks import decode_header
    from claudland import trigger as trg
    first, evs = _events(run_file, 60)
    dec = WaveformDecompressor()
    dec.load_constants(first)
    peds = [ev for ev in evs if trg.is_pedestal(decode_header(ev["Header"]).trigger_type)][:3]
    assert peds, "no pedestal events in the first records"
    pm_a = PedestalManager(); pm_b = PedestalManager()
    pm_a.set_compression_pedestals(dec.connection, dec.pedestals)
    pm_b.set_compression_pedestals(dec.connection, dec.pedestals)
    for ev in peds:
        b = dec.decompress_arrays(ev, "ID")
        assert pm_a.add_batch(b) == sum(pm_b.add(w) for w in b)
    assert pm_a.n_rejected == pm_b.n_rejected
    assert np.array_equal(pm_a._n, pm_b._n)
    assert np.allclose(pm_a._sum, pm_b._sum)
    # lookup: measured channels, compression-table fallback and out-of-range gain
    cable = np.array([0, 100, 1500, 1500, 2000, 5]); atwd = np.array([0, 1, 0, 1, 0, 0]); gain = np.array([0, 0, 1, 3, 0, 2])
    got = pm_a.get_batch(cable, atwd, gain)
    for i, (c, a, g) in enumerate(zip(cable, atwd, gain)):
        assert np.allclose(got[i], pm_a.get(int(c), int(a), int(g)))
