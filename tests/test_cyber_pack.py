from __future__ import annotations

import yaml


def test_cyber_questions_load_and_have_min_fields() -> None:
    with open("domain_packs/cyber/questions.yaml", "r", encoding="utf-8") as f:
        qs = yaml.safe_load(f)
    assert isinstance(qs, list)
    assert len(qs) >= 10  # aiming toward 12–15
    required = {"id", "text", "slot", "answer_values", "prior_p"}
    for q in qs:
        assert required.issubset(q.keys())


