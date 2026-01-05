"""
Constraint System for Procedural GPT.

Constraints C: S → {true, false} are logical predicates over symbolic states.
C(σ) = true iff σ satisfies all constraints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

from procedural_gpt.core.procedural_token import SymbolicState

S = TypeVar("S")


class Constraint(ABC, Generic[S]):
    """
    Abstract base class for constraints.

    A constraint is a logical predicate C: S → {true, false}.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this constraint."""
        pass

    @abstractmethod
    def check(self, state: SymbolicState[S]) -> bool:
        """
        Check if the state satisfies this constraint.

        Returns:
            True if constraint is satisfied, False otherwise.
        """
        pass

    @abstractmethod
    def to_vector(self) -> List[float]:
        """Convert constraint to numeric representation for embedding."""
        pass


@dataclass
class FunctionConstraint(Constraint[S]):
    """Constraint defined by a callable predicate function."""

    _name: str
    predicate: Callable[[SymbolicState[S]], bool]
    _vector: List[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self._name

    def check(self, state: SymbolicState[S]) -> bool:
        return self.predicate(state)

    def to_vector(self) -> List[float]:
        return self._vector if self._vector else [1.0]


@dataclass
class RangeConstraint(Constraint[S]):
    """Constraint that checks if a value is within a range."""

    _name: str
    key: str  # Key in state to check (for DictState)
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    @property
    def name(self) -> str:
        return self._name

    def check(self, state: SymbolicState[S]) -> bool:
        # Get value from state
        val = self._get_value(state)
        if val is None:
            return False

        if self.min_val is not None and val < self.min_val:
            return False
        if self.max_val is not None and val > self.max_val:
            return False
        return True

    def _get_value(self, state: SymbolicState[S]) -> Optional[float]:
        """Extract value from state based on key."""
        from procedural_gpt.core.procedural_token import IntegerState, DictState

        if isinstance(state, IntegerState):
            return float(state.value)
        elif isinstance(state, DictState):
            return state.data.get(self.key)
        else:
            vec = state.to_vector()
            return vec[0] if vec else None

    def to_vector(self) -> List[float]:
        return [
            self.min_val if self.min_val is not None else -1e6,
            self.max_val if self.max_val is not None else 1e6
        ]


class ConstraintSet(Generic[S]):
    """
    Collection of constraints that must all be satisfied.

    Implements C(σ) = ∧_c∈C c(σ)
    """

    def __init__(self, constraints: Optional[List[Constraint[S]]] = None):
        self._constraints: Dict[str, Constraint[S]] = {}
        if constraints:
            for c in constraints:
                self.add(c)

    def add(self, constraint: Constraint[S]) -> None:
        """Add a constraint to the set."""
        self._constraints[constraint.name] = constraint

    def remove(self, name: str) -> None:
        """Remove a constraint by name."""
        self._constraints.pop(name, None)

    def get(self, name: str) -> Optional[Constraint[S]]:
        """Get a constraint by name."""
        return self._constraints.get(name)

    def check(self, state: SymbolicState[S]) -> bool:
        """
        Check if state satisfies ALL constraints.

        Implements C(σ) = true iff σ satisfies all constraints.
        """
        return all(c.check(state) for c in self._constraints.values())

    def check_subset(self, state: SymbolicState[S], names: Set[str]) -> bool:
        """Check if state satisfies a subset of constraints."""
        for name in names:
            if name in self._constraints:
                if not self._constraints[name].check(state):
                    return False
        return True

    def get_violated(self, state: SymbolicState[S]) -> List[str]:
        """Get list of violated constraint names."""
        return [
            name for name, c in self._constraints.items()
            if not c.check(state)
        ]

    def to_vector(self) -> List[float]:
        """Convert all constraints to a single vector."""
        result = []
        for c in self._constraints.values():
            result.extend(c.to_vector())
        return result

    @property
    def names(self) -> Set[str]:
        """Get all constraint names."""
        return set(self._constraints.keys())

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        return iter(self._constraints.values())

    def __contains__(self, name: str) -> bool:
        return name in self._constraints
