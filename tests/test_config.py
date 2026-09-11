"""Configuration file handling and the run-list parser."""
import os
import tempfile

import numpy as np
import pytest

from claudland import config, runinfo


def test_defaults_and_search():
    cfg = config.get()
    assert cfg.pmt_table.is_absolute() and cfg.data_dir.is_absolute()
    assert cfg.pmt_table.name == "pmt_xyz.dat"


def test_explicit_file_and_missing_input():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "claudland.toml")
        with open(path, "w") as fh:
            fh.write('[paths]\npmt_table = "nowhere/pmt.dat"\ndata_dir = "data"\n[calibration]\ntq = "c.json"\n')
        cfg = config.load(path)
        assert cfg.source is not None and cfg.base == config.Path(d).resolve()
        assert str(cfg.pmt_table).endswith(os.path.join("nowhere", "pmt.dat"))
        assert cfg.run_info.name == "run-info.table"          # default filled in
        assert cfg.find_run(1) is None                         # data_dir does not exist
        try:
            cfg.require("pmt_table")
        except config.PrivateInputMissing as exc:
            assert "pmt" in str(exc)
        else:
            raise AssertionError("PrivateInputMissing not raised")
        assert cfg.optional("tq") is None


def test_runinfo_parser():
    with tempfile.NamedTemporaryFile("w", suffix=".table", delete=False) as fh:
        fh.write("000166   2002/03/05(Tue)02:22(57)   5.764   1015262577  0   1   OD inefficiency ~13%\n")
        fh.write("009937   2010/12/05(Sun)10:04(23)   1.576   1291511063  1   6  HV10 is down, 17 and 20 inch half badrun -> veto(141054016614 ~ end)\n")
        fh.write("garbage line\n")
        name = fh.name
    try:
        tab = runinfo.load(name)
    finally:
        os.unlink(name)
    assert len(tab) == 2 and tab["run"].tolist() == [166, 9937]
    assert abs(tab["hours"][0] - 5.764) < 1e-9
    assert tab["grade"].tolist() == [1, 6] and tab["flag"].tolist() == [0, 1]
    assert runinfo.veto_intervals(str(tab["comment"][1])) == [(141054016614, None)]
    assert runinfo.veto_intervals("nothing here") == []


def test_runinfo_from_config():
    if not config.get().run_info.exists():
        pytest.skip("private run-info.table not available")
    tab = runinfo.load()
    assert len(tab) > 10000
    row = runinfo.lookup(1467)
    assert row is not None and row["grade"] == 0
    assert 1467 in runinfo.good_runs(0)
