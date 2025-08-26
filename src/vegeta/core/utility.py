from __future__ import annotations

from math import inf
from typing import Callable, Dict, Iterable, Tuple, TypeVar


Action = TypeVar("Action")
State = TypeVar("State")
UtilityFunction = Callable[[Action, State], float]


def value_of_action(
    action: Action,
    belief_over_states: Dict[State, float],
    utility: UtilityFunction[Action, State],
) -> float:
    """Expected value of a specific action under a belief distribution."""

    return sum(prob * utility(action, state) for state, prob in belief_over_states.items())


def best_action(
    actions: Iterable[Action],
    belief_over_states: Dict[State, float],
    utility: UtilityFunction[Action, State],
) -> Tuple[Action, float]:
    """Return (action, value) that maximizes expected value under the belief."""

    best_val = -inf
    best_act: Action | None = None
    for action in actions:
        val = value_of_action(action, belief_over_states, utility)
        if val > best_val:
            best_val = val
            best_act = action
    assert best_act is not None  # actions must be non-empty
    return best_act, best_val


def expected_value(
    actions: Iterable[Action],
    belief_over_states: Dict[State, float],
    utility: UtilityFunction[Action, State],
) -> float:
    """Maximum expected value over the provided actions under the belief.

    This returns the value of the optimal action (argmax omitted for simplicity).
    """

    return best_action(actions, belief_over_states, utility)[1]


