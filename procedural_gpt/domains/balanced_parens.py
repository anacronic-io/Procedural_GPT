"""
Balanced Parentheses Domain.

Example from the paper (Section 3.5):
S = Z (nesting depth)
C(σ) = (σ ≥ 0)
T(σ, "(") = σ + 1
T(σ, ")") = σ - 1 if σ > 0, else ⊥

This domain ensures generated parentheses sequences are always balanced.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from procedural_gpt.core.procedural_token import SymbolicState, ProceduralToken
from procedural_gpt.core.transition import TransitionFunction, UNDEFINED, UndefinedTransition
from procedural_gpt.core.constraints import Constraint, ConstraintSet, RangeConstraint


@dataclass
class BalancedParensState(SymbolicState[int]):
    """
    Symbolic state for balanced parentheses.

    State is simply the nesting depth (integer).
    """

    depth: int
    max_depth: int = 10  # Maximum allowed depth

    def to_vector(self) -> List[float]:
        """Encode depth as normalized vector."""
        return [
            float(self.depth),
            float(self.depth) / self.max_depth,  # Normalized depth
            1.0 if self.depth == 0 else 0.0,     # At base level
            1.0 if self.depth >= self.max_depth else 0.0  # At max depth
        ]

    def clone(self) -> "BalancedParensState":
        return BalancedParensState(
            depth=self.depth,
            max_depth=self.max_depth
        )

    def __hash__(self) -> int:
        return hash((self.depth, self.max_depth))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BalancedParensState):
            return False
        return self.depth == other.depth and self.max_depth == other.max_depth


class DepthConstraint(Constraint[int]):
    """Constraint: depth must be non-negative and within max."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return "depth_valid"

    def check(self, state: SymbolicState[int]) -> bool:
        if isinstance(state, BalancedParensState):
            return 0 <= state.depth <= state.max_depth
        return True

    def to_vector(self) -> List[float]:
        return [float(self.max_depth)]


class BalancedParensTransition(TransitionFunction[int]):
    """
    Transition function for balanced parentheses.

    T(σ, "(") = σ + 1
    T(σ, ")") = σ - 1 if σ > 0, else ⊥
    """

    def __init__(
        self,
        max_depth: int = 10,
        include_content: bool = True
    ):
        """
        Initialize transition function.

        Args:
            max_depth: Maximum nesting depth
            include_content: Whether to include content tokens (letters)
        """
        # Build vocabulary
        vocab = ["(", ")", "<PAD>", "<BOS>", "<EOS>"]
        if include_content:
            # Add lowercase letters as content tokens
            vocab.extend(list("abcdefghijklmnopqrstuvwxyz"))
            vocab.extend(list(" "))

        # Build constraints
        constraints = ConstraintSet([
            DepthConstraint(max_depth)
        ])

        super().__init__(vocab, constraints)

        self.max_depth = max_depth
        self.include_content = include_content

    def transition(
        self,
        state: SymbolicState[int],
        token: str
    ) -> SymbolicState[int] | UndefinedTransition:
        """
        Apply transition.

        T(σ, "(") = σ + 1
        T(σ, ")") = σ - 1 if σ > 0, else ⊥
        """
        if not isinstance(state, BalancedParensState):
            return UNDEFINED

        if token == "(":
            # Opening paren: increment depth
            if state.depth >= state.max_depth:
                return UNDEFINED  # Would exceed max depth
            return BalancedParensState(
                depth=state.depth + 1,
                max_depth=state.max_depth
            )

        elif token == ")":
            # Closing paren: decrement depth
            if state.depth <= 0:
                return UNDEFINED  # Would go negative
            return BalancedParensState(
                depth=state.depth - 1,
                max_depth=state.max_depth
            )

        elif token in ("<PAD>", "<BOS>", "<EOS>"):
            # Special tokens don't change state
            return state.clone()

        elif self.include_content:
            # Content tokens don't change depth
            return state.clone()

        return UNDEFINED

    def can_close(self, state: BalancedParensState) -> bool:
        """Check if sequence can be closed (reach depth 0)."""
        return state.depth == 0


def create_balanced_parens_domain(
    max_depth: int = 10,
    include_content: bool = True
) -> Tuple[BalancedParensState, BalancedParensTransition, ConstraintSet]:
    """
    Create a balanced parentheses domain.

    Returns:
        Tuple of (initial_state, transition_function, constraints)
    """
    transition = BalancedParensTransition(
        max_depth=max_depth,
        include_content=include_content
    )

    initial_state = BalancedParensState(depth=0, max_depth=max_depth)

    return initial_state, transition, transition.constraints


# Utility functions for the domain

def generate_balanced_sequence(length: int) -> str:
    """Generate a random balanced parentheses sequence."""
    import random

    result = []
    depth = 0
    remaining = length

    while remaining > 0:
        if depth == 0:
            # Must open
            result.append("(")
            depth += 1
        elif remaining == depth:
            # Must close all
            result.append(")")
            depth -= 1
        else:
            # Can either open or close
            if random.random() < 0.5 and depth < length // 2:
                result.append("(")
                depth += 1
            else:
                result.append(")")
                depth -= 1
        remaining -= 1

    return "".join(result)


def is_balanced(sequence: str) -> bool:
    """Check if a parentheses sequence is balanced."""
    depth = 0
    for char in sequence:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def create_training_data(
    num_samples: int,
    min_length: int = 4,
    max_length: int = 20
) -> List[str]:
    """Generate training data of balanced sequences."""
    import random

    data = []
    for _ in range(num_samples):
        # Random even length
        length = random.randint(min_length // 2, max_length // 2) * 2
        data.append(generate_balanced_sequence(length))

    return data
