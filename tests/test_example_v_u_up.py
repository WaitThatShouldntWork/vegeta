from vegeta.core.utility import expected_value


def test_prior_vs_posterior_value():
    actions = ["A", "B"]

    def u(a: str, s: str) -> float:
        return 1.0 if a == s else 0.0

    v_prior = expected_value(actions, {"A": 0.6, "B": 0.4}, u)
    v_post = expected_value(actions, {"A": 0.3, "B": 0.7}, u)

    assert v_prior == 0.6
    assert v_post == 0.7


