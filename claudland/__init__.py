"""claudland -- a Python framework for reading and reconstructing KamLAND raw data.

Modules
-------
sf        Serial-format (``.sf`` / ``.sfz``) file reader
banks     Typed decoders for the Header, RunHeader, History and HitHeader banks
trigger   Trigger-type bit definitions
wfcomp    ATWD waveform decompression (Huffman codec of the ``.sfz`` files)
pedestal  Pedestal and baseline handling
tq        Pulse finding: time and charge per waveform
calib     Sampling period, 1 p.e. charge, time offsets
geometry  PMT positions and cable conventions
vertex    Time-of-flight vertex fitter
energy    Visible-energy estimators
reco      The reconstruction pipeline
"""
from .sf import SFReader, SFEvent, SFBank
from .banks import decode_header, decode_run_header, decode_hit_header, decode_history
from . import config
from .wfcomp import WaveformDecompressor, WaveformBatch, Waveform
from .reco import EventReconstructor, RecoEvent

__version__ = "0.1.0"
__all__ = ["config", "SFReader", "SFEvent", "SFBank", "WaveformDecompressor", "WaveformBatch", "Waveform", "EventReconstructor", "RecoEvent",
           "decode_header", "decode_run_header", "decode_hit_header", "decode_history"]
