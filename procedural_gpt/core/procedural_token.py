"""
Procedural Token: The core latent state variable.

Procedural Token p_t = (σ_t, R_t) where:
- σ_t ∈ S: Symbolic state (domain-specific)
- R_t ⊆ C: Active constraints (logical predicates)

Key Properties:
1. Internal: p_t ∉ V; never emitted to output
2. Executable: Determines valid future actions via A(σ_t)
3. Differentiable via E_p: Maps to continuous embeddings for gradient flow
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Generic, Optional, Set, TypeVar
import copy

# Type variable for domain-specific state
S = TypeVar("S")


class SymbolicState(ABC, Generic[S]):
    """
    Abstract base class for domain-specific symbolic states.

    Each domain must implement:
    - to_vector(): Convert state to a numeric representation for E_p
    - clone(): Create a deep copy of the state
    - __hash__ and __eq__: For caching and comparison
    """

    @abstractmethod
    def to_vector(self) -> list:
        """Convert symbolic state to numeric vector for neural encoding."""
        pass

    @abstractmethod
    def clone(self) -> "SymbolicState[S]":
        """Create a deep copy of this state."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for caching valid action sets."""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Equality for state comparison."""
        pass


@dataclass
class IntegerState(SymbolicState[int]):
    """Simple integer-valued symbolic state (e.g., for balanced parentheses depth)."""

    value: int

    def to_vector(self) -> list:
        return [float(self.value)]

    def clone(self) -> "IntegerState":
        return IntegerState(value=self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntegerState):
            return False
        return self.value == other.value


@dataclass
class DictState(SymbolicState[Dict[str, Any]]):
    """Dictionary-based symbolic state for complex domains."""

    data: Dict[str, Any] = field(default_factory=dict)
    _vector_keys: tuple = field(default_factory=tuple)

    def to_vector(self) -> list:
        """Convert state dict to vector using specified keys."""
        result = []
        for key in self._vector_keys:
            val = self.data.get(key, 0)
            if isinstance(val, (int, float)):
                result.append(float(val))
            elif isinstance(val, bool):
                result.append(1.0 if val else 0.0)
            elif isinstance(val, (list, tuple)):
                result.extend([float(v) for v in val])
            else:
                result.append(0.0)
        return result

    def clone(self) -> "DictState":
        return DictState(
            data=copy.deepcopy(self.data),
            _vector_keys=self._vector_keys
        )

    def __hash__(self) -> int:
        # Convert dict to hashable representation
        items = tuple(sorted((k, str(v)) for k, v in self.data.items()))
        return hash(items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictState):
            return False
        return self.data == other.data


@dataclass
class ProceduralToken(Generic[S]):
    """
    Procedural Token: p_t = (σ_t, R_t)

    The fundamental latent state variable that conditions generation.

    Attributes:
        sigma: The symbolic state σ_t ∈ S
        R: Active constraints R_t ⊆ C (as frozen set of constraint IDs)
        step: Current generation step t
    """

    sigma: SymbolicState[S]
    R: FrozenSet[str] = field(default_factory=frozenset)
    step: int = 0

    def clone(self) -> "ProceduralToken[S]":
        """Create a deep copy of this procedural token."""
        return ProceduralToken(
            sigma=self.sigma.clone(),
            R=self.R,  # FrozenSet is immutable, no need to copy
            step=self.step
        )

    def with_state(self, new_sigma: SymbolicState[S]) -> "ProceduralToken[S]":
        """Return new token with updated state."""
        return ProceduralToken(
            sigma=new_sigma,
            R=self.R,
            step=self.step + 1
        )

    def with_constraints(self, new_R: FrozenSet[str]) -> "ProceduralToken[S]":
        """Return new token with updated constraints."""
        return ProceduralToken(
            sigma=self.sigma.clone(),
            R=new_R,
            step=self.step
        )

    def __hash__(self) -> int:
        return hash((self.sigma, self.R, self.step))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProceduralToken):
            return False
        return (
            self.sigma == other.sigma
            and self.R == other.R
            and self.step == other.step
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sigma_vector": self.sigma.to_vector(),
            "constraints": list(self.R),
            "step": self.step
        }
