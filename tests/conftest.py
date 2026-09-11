import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from claudland import config as _config

_RUN = _config.get().find_run(2279)
RUN_FILE = str(_RUN) if _RUN else os.path.join(ROOT, "..", "Run", "co60-zscan", "run_002279_000000_000001.sfz")


@pytest.fixture(scope="session")
def run_file():
    if not os.path.exists(RUN_FILE):
        pytest.skip("run 2279 data file not available (see claudland.toml: data_dir)")
    return RUN_FILE
