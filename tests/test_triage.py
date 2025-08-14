from __future__ import annotations

from decider.triage import recommend_action


def test_triage_recommends_patch_now_when_high_risk() -> None:
    action, details = recommend_action(
        cvss=9.0, epss=0.6, kev_flag=True, asset_present=True, internet_exposed=False, business_critical=False
    )
    assert action == "patch_now"
    assert 0.0 <= details["risk"] <= 1.0


def test_triage_escalates_on_exposure() -> None:
    action, _ = recommend_action(
        cvss=5.0, epss=0.2, kev_flag=False, asset_present=True, internet_exposed=True, business_critical=False
    )
    assert action in {"plan_patch", "patch_now"}


