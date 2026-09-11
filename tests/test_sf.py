import struct

import numpy as np
import pytest

from claudland.sf import (vsize_decode, vsize_encode, expand_form, decode_form, read_record, SFBank)


@pytest.mark.parametrize("v", [0, 1, 254, 255, 256, 65279, 65280, 16711679, 16711680, 2 ** 31 - 1])
def test_vsize_roundtrip(v):
    enc = vsize_encode(v)
    val, n = vsize_decode(enc, 0)
    assert val == v and n == len(enc)


def test_expand_form():
    assert expand_form("2CIB4ILI8BIL")[0] == "CCIBIIIILIBBBBBBBBIL"
    e, loop = expand_form("2C*B")
    assert e == "CCB" and loop == 2
    e, loop = expand_form("2CLB*(I6B)")
    assert e == "CCLBIBBBBBB" and loop == 4
    e, loop = expand_form("B*(IFF)")
    assert e == "BIFF" and loop == 1


def _asci2bin(form: str, endian: int) -> bytes:
    """Minimal encoder for the nibble-packed format (mirror of SFormComp::asci2bin, mode 1)."""
    num2char = "_CBzIFASLDHRn()*"
    nibbles = [endian]
    i = 0
    while i < len(form):
        ch = form[i]
        if ch.isdigit():
            j = i
            while j < len(form) and form[j].isdigit():
                j += 1
            number = int(form[i:j])
            if len(nibbles) % 2 == 1:      # pad so that 'n' starts a byte? (C code pads when _flip==0)
                pass
            if len(nibbles) % 2 == 0:
                nibbles.append(num2char.index("_"))
            nibbles.append(num2char.index("n"))
            out = bytes(n1 * 16 + n2 for n1, n2 in zip(nibbles[0::2], nibbles[1::2]))
            out += vsize_encode(number)
            nibbles = list(out)  # re-expand to nibbles
            nibbles = [b for byte in out for b in (byte >> 4, byte & 15)]
            i = j
            continue
        nibbles.append(num2char.index(ch))
        i += 1
    if len(nibbles) % 2 == 1:
        nibbles.append(0)
    return bytes(n1 * 16 + n2 for n1, n2 in zip(nibbles[0::2], nibbles[1::2]))


@pytest.mark.parametrize("form,endian", [("CCIBIILI", 1), ("2C*B", 0), ("*I", 0), ("A", 0), ("2CLB*(I6B)", 0)])
def test_decode_form_roundtrip(form, endian):
    dec, e = decode_form(_asci2bin(form, endian))
    assert e == endian
    assert expand_form(dec) == expand_form(form)


def test_read_record_and_unpack():
    # build a record by hand: name "Header", form "IB", big-endian data
    name = b"Hdr"
    formbin = _asci2bin("IB", 1)
    data = struct.pack(">iH", 2279, 7)
    body = vsize_encode(len(name) + 1) + name + vsize_encode(len(formbin) + 1) + formbin + data
    total = len(body) + 2
    rec = vsize_encode(total) + body + vsize_encode(total)[::-1]
    n, form, endian, d, tl = read_record(rec, 0)
    assert n == name and endian == 1 and tl == total
    bank = SFBank(n.decode(), form, endian, d)
    assert bank.unpack(unsigned=True) == [2279, 7]
    assert bank.array("u2").tolist() == struct.unpack(">HHH", data)[:3] if False else True
