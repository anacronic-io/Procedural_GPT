"""
Transition Function for Procedural GPT.

T: S × V ⇀ S (partial function)
where T(σ, τ) = ⊥ indicates undefined transition.

The transition function is the core symbolic component that:
1. Updates state based on emitted tokens
2. Determines valid action sets A(σ)
3. Provides formal correctness guarantees
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

from procedural_gpt.core.procedural_token import ProceduralToken, SymbolicState
from procedural_gpt.core.constraints import ConstraintSet

S = TypeVar("S")


# Sentinel value for undefined transitions
class UndefinedTransition:
    """Sentinel class representing T(σ, τ) = ⊥ (undefined transition)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "⊥"

    def __bool__(self):
        return False


UNDEFINED = UndefinedTransition()


class TransitionFunction(ABC, Generic[S]):
    """
    Abstract base class for transition functions.

    T: S × V ⇀ S

    Subclasses must implement:
    - transition(): Compute T(σ, τ)
    - vocabulary: The set of all possible tokens V
    """

    def __init__(
        self,
        vocabulary: List[str],
        constraints: Optional[ConstraintSet[S]] = None
    ):
        """
        Initialize transition function.

        Args:
            vocabulary: List of all tokens V
            constraints: ConstraintSet for checking validity
        """
        self._vocabulary = vocabulary
        self._vocab_set = set(vocabulary)
        self._vocab_to_idx = {v: i for i, v in enumerate(vocabulary)}
        self._constraints = constraints or ConstraintSet()

        # Cache for valid action sets: state_hash -> set of valid tokens
        self._action_cache: Dict[int, Set[str]] = {}

    @abstractmethod
    def transition(
        self,
        state: SymbolicState[S],
        token: str
    ) -> SymbolicState[S] | UndefinedTransition:
        """
        Compute T(σ, τ).

        Args:
            state: Current symbolic state σ
            token: Token to apply τ

        Returns:
            New state σ' = T(σ, τ), or UNDEFINED if transition is invalid.
        """
        pass

    def is_valid_transition(self, state: SymbolicState[S], token: str) -> bool:
        """
        Check if T(σ, τ) ≠ ⊥ and C(T(σ, τ)) = true.
        """
        if token not in self._vocab_set:
            return False

        new_state = self.transition(state, token)
        if new_state is UNDEFINED:
            return False

        return self._constraints.check(new_state)

    def valid_actions(
        self,
        state: SymbolicState[S],
        use_cache: bool = True
    ) -> Set[str]:
        """
        Compute A(σ) = {v ∈ V | T(σ, v) ≠ ⊥ and C(T(σ, v)) = true}.

        Args:
            state: Current symbolic state
            use_cache: Whether to use/update cache

        Returns:
            Set of valid tokens at this state.
        """
        state_hash = hash(state)

        if use_cache and state_hash in self._action_cache:
            return self._action_cache[state_hash]

        valid = set()
        for token in self._vocabulary:
            if self.is_valid_transition(state, token):
                valid.add(token)

        if use_cache:
            self._action_cache[state_hash] = valid

        return valid

    def valid_action_mask(
        self,
        state: SymbolicState[S],
        use_cache: bool = True
    ) -> List[bool]:
        """
        Get boolean mask indicating valid actions.

        Returns:
            List of booleans, True if token at that index is valid.
        """
        valid = self.valid_actions(state, use_cache)
        return [token in valid for token in self._vocabulary]

    def apply(
        self,
        procedural_token: ProceduralToken[S],
        token: str
    ) -> Optional[ProceduralToken[S]]:
        """
        Apply transition to procedural token, returning updated token.

        Returns:
            New procedural token with updated state, or None if invalid.
        """
        new_state = self.transition(procedural_token.sigma, token)
        if new_state is UNDEFINED:
            return None

        if not self._constraints.check(new_state):
            return None

        return procedural_token.with_state(new_state)

    def clear_cache(self) -> None:
        """Clear the valid action cache."""
        self._action_cache.clear()

    @property
    def vocabulary(self) -> List[str]:
        return self._vocabulary

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def constraints(self) -> ConstraintSet[S]:
        return self._constraints

    def token_to_idx(self, token: str) -> int:
        """Convert token to vocabulary index."""
        return self._vocab_to_idx.get(token, -1)

    def idx_to_token(self, idx: int) -> str:
        """Convert vocabulary index to token."""
        if 0 <= idx < len(self._vocabulary):
            return self._vocabulary[idx]
        raise IndexError(f"Index {idx} out of vocabulary range")


@dataclass
class TransitionRule:
    """A single transition rule for rule-based transitions."""

    condition: Callable[[SymbolicState, str], bool]
    action: Callable[[SymbolicState, str], SymbolicState]
    priority: int = 0


class RuleBasedTransition(TransitionFunction[S]):
    """
    Transition function defined by a set of rules.

    Rules are checked in priority order; first matching rule is applied.
    """

    def __init__(
        self,
        vocabulary: List[str],
        rules: List[TransitionRule],
        constraints: Optional[ConstraintSet[S]] = None
    ):
        super().__init__(vocabulary, constraints)
        self._rules = sorted(rules, key=lambda r: -r.priority)

    def transition(
        self,
        state: SymbolicState[S],
        token: str
    ) -> SymbolicState[S] | UndefinedTransition:
        for rule in self._rules:
            if rule.condition(state, token):
                try:
                    return rule.action(state, token)
                except Exception:
                    return UNDEFINED
        return UNDEFINED

    def add_rule(self, rule: TransitionRule) -> None:
        """Add a new rule to the transition function."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)
        self.clear_cache()
