from vegeta.core.utility import expected_value


def test_expected_value_binary_choice():
    belief = {"A": 0.6, "B": 0.4}
    actions = ["A", "B"]

    def u(a: str, s: str) -> float:
        return 1.0 if a == s else 0.0

    v = expected_value(actions, belief, u)
    assert v == 0.6


