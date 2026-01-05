"""
Tests for core Procedural GPT components.
"""

import pytest
import torch

from procedural_gpt.core.procedural_token import (
    ProceduralToken,
    IntegerState,
    DictState,
)
from procedural_gpt.core.transition import TransitionFunction, UNDEFINED
from procedural_gpt.core.constraints import (
    Constraint,
    ConstraintSet,
    FunctionConstraint,
    RangeConstraint,
)


class TestSymbolicState:
    """Tests for SymbolicState implementations."""

    def test_integer_state_to_vector(self):
        state = IntegerState(value=5)
        vec = state.to_vector()
        assert vec == [5.0]

    def test_integer_state_clone(self):
        state = IntegerState(value=5)
        cloned = state.clone()
        assert cloned.value == 5
        assert cloned is not state

    def test_integer_state_hash(self):
        state1 = IntegerState(value=5)
        state2 = IntegerState(value=5)
        state3 = IntegerState(value=6)
        assert hash(state1) == hash(state2)
        assert hash(state1) != hash(state3)

    def test_dict_state_to_vector(self):
        state = DictState(
            data={"x": 1, "y": 2, "flag": True},
            _vector_keys=("x", "y", "flag")
        )
        vec = state.to_vector()
        assert vec == [1.0, 2.0, 1.0]

    def test_dict_state_clone(self):
        state = DictState(data={"x": 1}, _vector_keys=("x",))
        cloned = state.clone()
        assert cloned.data == {"x": 1}
        cloned.data["x"] = 2
        assert state.data["x"] == 1  # Original unchanged


class TestProceduralToken:
    """Tests for ProceduralToken."""

    def test_create_token(self):
        state = IntegerState(value=0)
        token = ProceduralToken(sigma=state, R=frozenset(["c1"]), step=0)
        assert token.sigma.value == 0
        assert "c1" in token.R
        assert token.step == 0

    def test_clone(self):
        state = IntegerState(value=5)
        token = ProceduralToken(sigma=state, R=frozenset(["c1"]), step=3)
        cloned = token.clone()
        assert cloned.sigma.value == 5
        assert cloned.step == 3
        assert cloned is not token

    def test_with_state(self):
        state1 = IntegerState(value=0)
        token1 = ProceduralToken(sigma=state1, step=0)

        state2 = IntegerState(value=1)
        token2 = token1.with_state(state2)

        assert token2.sigma.value == 1
        assert token2.step == 1  # Incremented
        assert token1.sigma.value == 0  # Original unchanged

    def test_to_dict(self):
        state = IntegerState(value=5)
        token = ProceduralToken(sigma=state, R=frozenset(["c1"]), step=2)
        d = token.to_dict()
        assert d["sigma_vector"] == [5.0]
        assert "c1" in d["constraints"]
        assert d["step"] == 2


class TestConstraints:
    """Tests for constraint system."""

    def test_function_constraint(self):
        constraint = FunctionConstraint(
            _name="positive",
            predicate=lambda s: s.value > 0 if hasattr(s, 'value') else True
        )

        state_valid = IntegerState(value=5)
        state_invalid = IntegerState(value=-1)

        assert constraint.check(state_valid)
        assert not constraint.check(state_invalid)

    def test_range_constraint(self):
        constraint = RangeConstraint(
            _name="in_range",
            key="value",
            min_val=0,
            max_val=10
        )

        state_valid = IntegerState(value=5)
        state_too_low = IntegerState(value=-1)
        state_too_high = IntegerState(value=15)

        assert constraint.check(state_valid)
        assert not constraint.check(state_too_low)
        assert not constraint.check(state_too_high)

    def test_constraint_set(self):
        c1 = FunctionConstraint(
            _name="non_negative",
            predicate=lambda s: s.value >= 0 if hasattr(s, 'value') else True
        )
        c2 = RangeConstraint(_name="max_10", key="value", max_val=10)

        constraint_set = ConstraintSet([c1, c2])

        assert constraint_set.check(IntegerState(value=5))
        assert not constraint_set.check(IntegerState(value=-1))
        assert not constraint_set.check(IntegerState(value=15))

    def test_constraint_set_get_violated(self):
        c1 = FunctionConstraint(
            _name="positive",
            predicate=lambda s: s.value > 0 if hasattr(s, 'value') else True
        )
        c2 = FunctionConstraint(
            _name="less_than_10",
            predicate=lambda s: s.value < 10 if hasattr(s, 'value') else True
        )

        constraint_set = ConstraintSet([c1, c2])

        violated = constraint_set.get_violated(IntegerState(value=15))
        assert "less_than_10" in violated
        assert "positive" not in violated


class TestTransitionFunction:
    """Tests for transition function."""

    def test_valid_actions(self):
        from procedural_gpt.domains.balanced_parens import BalancedParensTransition, BalancedParensState

        transition = BalancedParensTransition(max_depth=5, include_content=False)

        # At depth 0, can only open
        state = BalancedParensState(depth=0, max_depth=5)
        valid = transition.valid_actions(state)
        assert "(" in valid
        assert ")" not in valid or state.depth > 0  # Can't close at depth 0

        # At depth > 0, can open or close
        state = BalancedParensState(depth=2, max_depth=5)
        valid = transition.valid_actions(state)
        assert "(" in valid
        assert ")" in valid

        # At max depth, can only close
        state = BalancedParensState(depth=5, max_depth=5)
        valid = transition.valid_actions(state)
        assert "(" not in valid  # Can't open at max depth
        assert ")" in valid

    def test_transition(self):
        from procedural_gpt.domains.balanced_parens import BalancedParensTransition, BalancedParensState

        transition = BalancedParensTransition(max_depth=5, include_content=False)
        state = BalancedParensState(depth=0, max_depth=5)

        # Valid transition
        new_state = transition.transition(state, "(")
        assert new_state.depth == 1

        # Invalid transition
        result = transition.transition(state, ")")
        assert result is UNDEFINED

    def test_caching(self):
        from procedural_gpt.domains.balanced_parens import BalancedParensTransition, BalancedParensState

        transition = BalancedParensTransition(max_depth=5, include_content=False)
        state = BalancedParensState(depth=2, max_depth=5)

        # First call - computes
        valid1 = transition.valid_actions(state, use_cache=True)

        # Second call - from cache
        valid2 = transition.valid_actions(state, use_cache=True)

        assert valid1 == valid2


class TestTheorem1Invariance:
    """Tests for Theorem 1: Operational Invariance."""

    def test_invariance_maintained(self):
        """
        Theorem 1: Under correct T and C specification,
        all generated states satisfy constraints.
        """
        from procedural_gpt.domains.balanced_parens import (
            BalancedParensTransition,
            BalancedParensState,
        )

        transition = BalancedParensTransition(max_depth=10, include_content=False)

        # Start from valid state
        state = BalancedParensState(depth=0, max_depth=10)
        assert transition.constraints.check(state)  # Initial valid

        # Generate sequence using only valid actions
        sequence = ["(", "(", ")", "(", ")", ")"]

        for token in sequence:
            # Verify token is in valid actions
            valid = transition.valid_actions(state)
            assert token in valid, f"Token {token} not in valid actions at depth {state.depth}"

            # Apply transition
            state = transition.transition(state, token)
            assert state is not UNDEFINED

            # Verify constraint maintained
            assert transition.constraints.check(state), f"Constraint violated at depth {state.depth}"

    def test_corollary_zero_violations(self):
        """
        Corollary: Zero constraint violations when only valid actions are taken.
        """
        from procedural_gpt.domains.balanced_parens import (
            BalancedParensTransition,
            BalancedParensState,
        )
        import random

        transition = BalancedParensTransition(max_depth=10, include_content=False)

        # Run multiple random generations
        for _ in range(100):
            state = BalancedParensState(depth=0, max_depth=10)
            violations = 0

            for _ in range(50):  # Generate up to 50 tokens
                valid = transition.valid_actions(state)
                if not valid:
                    break

                # Random valid action
                token = random.choice(list(valid))

                # Apply and check
                state = transition.transition(state, token)
                if not transition.constraints.check(state):
                    violations += 1

            assert violations == 0, "Zero violations should be guaranteed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
