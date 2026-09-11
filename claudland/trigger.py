"""Trigger-type bit definitions for the SF ``Header`` bank (AKatDefs.hh, ESFTriggerType).

Bits 4..7 form a 4-bit *ATWD acquisition* field (``0x10`` forced acq. A,
``0x20`` forced acq. B, ``0x30`` clock A, ``0x40`` clock B, ``0x50``/``0x60``
test pulse, ``0x70``/``0x80`` pedestal A/B, ``0x90``..``0xb0`` test-pulse
variants); bits 12..15 form the *calibration source* field.  The remaining
bits are independent flags.
"""
from __future__ import annotations

ID_GLOBAL = 0x1
ID_HISTORY = 0x2
FIVE_INCH = 0x4
ONE_PPS = 0x8
ACQ_MASK = 0xF0
FORCED_ACQ_A = 0x10
FORCED_ACQ_B = 0x20
CLOCK_A = 0x30
CLOCK_B = 0x40
TEST_PULSE_A = 0x50
TEST_PULSE_B = 0x60
PEDESTAL_A = 0x70
PEDESTAL_B = 0x80
TEST_PULSE_FORCED_A = 0x90
TEST_PULSE_FORCED_B = 0xA0
TEST_PULSE_NO_ACQ = 0xB0
PRESCALE = 0x100
GPS = 0x200
CALIBRATION_DELAYED = 0x400
SUPERNOVA = 0x800
CALIB_SOURCE_MASK = 0xF000
MACRO_SINGLES = 0x10000
MACRO_COINCIDENCE = 0x20000
MACRO_RANDOM = 0x40000
MACRO_MUON = 0x80000
OD_TOP = 0x100000
OD_UPPER = 0x200000
OD_LOWER = 0x400000
OD_BOTTOM = 0x800000
DELAYED = 0x1000000
PROMPT = 0x2000000
OD_TO_ID = 0x4000000
ID_TO_OD = 0x8000000
OD_HISTORY_TOP = 0x10000000
OD_HISTORY_UPPER = 0x20000000
OD_HISTORY_LOWER = 0x40000000
OD_HISTORY_BOTTOM = 0x80000000

_ACQ_NAMES = {
    0x10: "ForcedAcqA", 0x20: "ForcedAcqB", 0x30: "ClockA", 0x40: "ClockB",
    0x50: "TestPulseA", 0x60: "TestPulseB", 0x70: "PedestalA", 0x80: "PedestalB",
    0x90: "TestPulseForcedA", 0xA0: "TestPulseForcedB", 0xB0: "TestPulseNoAcq",
}
_FLAG_NAMES = [
    (ID_GLOBAL, "IDGlobal"), (ID_HISTORY, "IDHistory"), (FIVE_INCH, "5inch"),
    (ONE_PPS, "1PPS"), (PRESCALE, "Prescale"), (GPS, "GPS"),
    (CALIBRATION_DELAYED, "CalibDelayed"), (SUPERNOVA, "Supernova"),
    (MACRO_SINGLES, "MacroSingles"), (MACRO_COINCIDENCE, "MacroCoinc"), (MACRO_RANDOM, "MacroRandom"),
    (MACRO_MUON, "MacroMuon"), (OD_TOP, "ODTop"), (OD_UPPER, "ODUpper"),
    (OD_LOWER, "ODLower"), (OD_BOTTOM, "ODBottom"), (DELAYED, "Delayed"),
    (PROMPT, "Prompt"), (OD_TO_ID, "ODtoID"), (ID_TO_OD, "IDtoOD"),
    (OD_HISTORY_TOP, "ODHistTop"), (OD_HISTORY_UPPER, "ODHistUpper"),
    (OD_HISTORY_LOWER, "ODHistLower"), (OD_HISTORY_BOTTOM, "ODHistBottom"),
]


def acquisition_type(trigger_type: int) -> int:
    """The 4-bit ATWD acquisition field (``0`` if none)."""
    return trigger_type & ACQ_MASK


def is_pedestal(trigger_type: int) -> bool:
    """True for pedestal-trigger events (acquisition type PedestalA/B)."""
    return acquisition_type(trigger_type) in (PEDESTAL_A, PEDESTAL_B)


def is_forced_acquisition(trigger_type: int) -> bool:
    """True for forced-acquisition events (acquisition type ForcedAcqA/B)."""
    return acquisition_type(trigger_type) in (FORCED_ACQ_A, FORCED_ACQ_B)


def is_clock(trigger_type: int) -> bool:
    """True for clock-trigger events (acquisition type ClockA/B)."""
    return acquisition_type(trigger_type) in (CLOCK_A, CLOCK_B)


def is_test_pulse(trigger_type: int) -> bool:
    """True for test-pulse events."""
    return acquisition_type(trigger_type) in (TEST_PULSE_A, TEST_PULSE_B, TEST_PULSE_FORCED_A,
                                              TEST_PULSE_FORCED_B, TEST_PULSE_NO_ACQ)


def is_physics(trigger_type: int) -> bool:
    """A "normal" self-triggered event: ID global / prescale / OD / prompt / delayed.

    Excludes pedestal, clock, test-pulse and forced-acquisition events (whose
    waveforms contain no physics pulses) and pure GPS / 1PPS timing events.
    Follows ``TKLSFTriggerTypeScanner::IsAGlobalTrigger`` (AKat).
    """
    trigger_type &= 0xFFFFFFFF
    if acquisition_type(trigger_type) != 0:
        return False
    physics_bits = (ID_GLOBAL | FIVE_INCH | PRESCALE | CALIBRATION_DELAYED | CALIB_SOURCE_MASK
                    | OD_TOP | OD_UPPER | OD_LOWER | OD_BOTTOM | DELAYED | PROMPT | OD_TO_ID | ID_TO_OD)
    return bool(trigger_type & physics_bits)


def describe(trigger_type: int) -> str:
    """Human readable list of set trigger bits, e.g. ``'IDGlobal|Prescale'``."""
    parts = [name for bit, name in _FLAG_NAMES if trigger_type & bit]
    acq = acquisition_type(trigger_type)
    if acq:
        parts.append(_ACQ_NAMES.get(acq, f"Acq0x{acq:x}"))
    src = trigger_type & CALIB_SOURCE_MASK
    if src:
        parts.append(f"CalibSource0x{src:x}")
    rest = (trigger_type & 0xFFFFFFFF) & ~(ACQ_MASK | CALIB_SOURCE_MASK | sum(b for b, _ in _FLAG_NAMES))
    if rest:
        parts.append(f"0x{rest:x}")
    return "|".join(parts) if parts else "none"
