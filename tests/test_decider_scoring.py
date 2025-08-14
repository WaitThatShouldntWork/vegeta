from __future__ import annotations

from decider.confidence import compute_answer_confidence
from decider.eig import score_answer, score_ask, score_search


def test_confidence_bounds() -> None:
    c = compute_answer_confidence({"cvss_norm": 2.0, "epss": -1.0, "kev_flag": 10})
    assert 0.0 <= c <= 1.0


def test_scoring_shapes() -> None:
    ans = score_answer(0.8)
    ask = score_ask(1.0)
    sea = score_search(recency_flag=True)
    for d in (ans, ask, sea):
        assert set(d.keys()) == {"eig", "cost", "net"}


