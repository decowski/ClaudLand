import numpy as np

from claudland.tq import fast_tq, multi_tq
from claudland.pedestal import baseline, baseline_batch


def _pulse(t0, a, t):
    return a * np.exp(-(t - t0) / 6.0) * (1 - np.exp(-(t - t0) / 1.5)) * (t >= t0)


def test_fast_vs_multi():
    rng = np.random.default_rng(1)
    t = np.arange(128.0)
    single = _pulse(30, 40, t)
    double = single + _pulse(70, 25, t)
    W = np.stack([single, double, np.zeros(128)]) + rng.normal(0, 1.0, (3, 128))
    r = fast_tq(W)
    assert r["found"].tolist() == [True, True, False]
    assert r["npulse"].tolist()[:2] == [1, 2]
    for i in range(2):
        pulses, tot, wf = multi_tq(W[i])
        assert abs(r["q_total"][i] - tot) < 0.05 * tot
        assert abs(r["t_lead"][i] - pulses[0].lead) <= 2
    assert abs(r["q_total"][0] - single.sum()) < 0.1 * single.sum()
    assert 29 < r["t_cfd"][0] < 33


def test_baseline():
    rng = np.random.default_rng(2)
    w = rng.normal(5.0, 1.0, 128)
    w[50:70] += 80.0
    assert abs(baseline(w) - 5.0) < 0.5
    b = baseline_batch(np.stack([w, w - 3]))
    assert abs(b[0] - 5.0) < 0.5 and abs(b[1] - 2.0) < 0.5
