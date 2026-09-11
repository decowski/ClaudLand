"""Round-trip test of the Huffman waveform codec against a Python port of the encoder."""
import numpy as np
import pytest

from claudland import huffman_tables
if not huffman_tables.available():
    pytest.skip("private Huffman tables not available (see claudland.toml)", allow_module_level=True)
NBITS, CODES = huffman_tables.NBITS, huffman_tables.CODES
from claudland.wfcomp import HuffmanWaveformDecoder, decode_thorsten, THORSTEN_BITS, NSAMPLES


def enc_d(sarray):
    """Forward difference transform (port of enc_d in DiffEntropyHuffmanCoDec.cc)."""
    d = [0] * 128
    dold = 18
    for m in range(127, -1, -1):
        d[m] = (sarray[m] - dold) & 1023
        dold = sarray[m]
    dold = 0
    for m in range(127, -1, -1):
        dsav = d[m]
        if (dold & 512) == 0:
            d[m] = (-d[m]) & 1023
        dold = dsav
    dsav = 0
    for m in range(127, -1, -1):
        dold = d[m]
        if ((dsav + 2) & 1023) > 4:
            shift = dsav + 2 if (dsav & 512) else dsav - 2
            d[m] = (d[m] - shift) & 1023
        dsav = dold
    return d


def encode(symbols):
    """Pack Huffman codes MSB-first into little-endian 32-bit words (port of Bitter::sendBits)."""
    bits = "".join(format(CODES[s], f"0{NBITS[s]}b") for s in symbols)
    nbits = len(bits)
    pad = (-nbits) % 32
    bits += "0" * pad
    words = [int(bits[i:i + 32], 2) for i in range(0, len(bits), 32)]
    return np.array(words, dtype="<u4").tobytes(), len(words) * 32


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_huffman_roundtrip(seed):
    rng = np.random.default_rng(seed)
    ped = rng.integers(100, 300, NSAMPLES).astype(np.uint16)
    wave = (ped.astype(int) + rng.normal(0, 2, NSAMPLES).round().astype(int)).clip(0, 1023)
    # add a pulse
    t = np.arange(NSAMPLES)
    wave = (wave + (60 * np.exp(-(t - 40) / 8.0) * (t >= 40)).astype(int)).clip(0, 1023)
    syms = enc_d((wave - ped.astype(int)).tolist())
    code, nbits = encode(syms)
    out, used = HuffmanWaveformDecoder.decode_symbols(code, nbits)
    assert out == syms
    assert nbits - 32 < used <= nbits
    rec = HuffmanWaveformDecoder.decode(code, nbits, ped)
    assert np.array_equal(rec, wave)


def test_thorsten_roundtrip():
    rng = np.random.default_rng(5)
    wave = rng.integers(0, 1024, NSAMPLES)
    words = np.zeros(43, dtype=np.uint32)
    for i in range(NSAMPLES):
        words[42 - i // 3] |= np.uint32(int(wave[i]) << (10 * (i % 3)))
    assert np.array_equal(decode_thorsten(words.astype("<u4").tobytes()), wave)
    assert THORSTEN_BITS == 43 * 32
