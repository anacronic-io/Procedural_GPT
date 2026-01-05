"""
Type-Safe Python Generation Domain.

This domain generates Python code that passes mypy type checking.
State tracks:
- Variable declarations and their types
- Scope stack
- Expected return types
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from procedural_gpt.core.procedural_token import SymbolicState, DictState
from procedural_gpt.core.transition import TransitionFunction, UNDEFINED, UndefinedTransition
from procedural_gpt.core.constraints import Constraint, ConstraintSet


class PythonType(Enum):
    """Basic Python types."""
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    NONE = "None"
    ANY = "Any"


@dataclass
class Scope:
    """A scope containing variable bindings."""
    variables: Dict[str, PythonType] = field(default_factory=dict)
    return_type: Optional[PythonType] = None
    is_function: bool = False


@dataclass
class TypedPythonState(SymbolicState[Dict]):
    """
    Symbolic state for type-safe Python generation.

    Tracks:
    - scope_stack: Stack of scopes with variable bindings
    - current_expr_type: Type of current expression being built
    - indent_level: Current indentation
    - in_assignment: Whether we're in an assignment context
    - expected_type: Expected type for current expression
    """

    scope_stack: List[Scope] = field(default_factory=lambda: [Scope()])
    current_expr_type: Optional[PythonType] = None
    indent_level: int = 0
    in_assignment: bool = False
    target_var: Optional[str] = None
    expected_type: Optional[PythonType] = None

    def to_vector(self) -> List[float]:
        """Encode state as vector."""
        # Encode scope depth
        scope_depth = len(self.scope_stack)

        # Encode current scope variable count
        var_count = len(self.scope_stack[-1].variables) if self.scope_stack else 0

        # Encode types as one-hot
        type_encoding = [0.0] * len(PythonType)
        if self.current_expr_type:
            type_encoding[list(PythonType).index(self.current_expr_type)] = 1.0

        expected_encoding = [0.0] * len(PythonType)
        if self.expected_type:
            expected_encoding[list(PythonType).index(self.expected_type)] = 1.0

        return [
            float(scope_depth),
            float(var_count),
            float(self.indent_level),
            1.0 if self.in_assignment else 0.0,
        ] + type_encoding + expected_encoding

    def clone(self) -> "TypedPythonState":
        return TypedPythonState(
            scope_stack=[
                Scope(
                    variables=dict(s.variables),
                    return_type=s.return_type,
                    is_function=s.is_function
                )
                for s in self.scope_stack
            ],
            current_expr_type=self.current_expr_type,
            indent_level=self.indent_level,
            in_assignment=self.in_assignment,
            target_var=self.target_var,
            expected_type=self.expected_type
        )

    def __hash__(self) -> int:
        # Simplified hash for caching
        var_tuple = tuple(
            (k, v.value)
            for s in self.scope_stack
            for k, v in sorted(s.variables.items())
        )
        return hash((
            var_tuple,
            self.current_expr_type,
            self.indent_level,
            self.in_assignment
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedPythonState):
            return False
        return (
            len(self.scope_stack) == len(other.scope_stack)
            and self.current_expr_type == other.current_expr_type
            and self.indent_level == other.indent_level
            and self.in_assignment == other.in_assignment
        )

    @property
    def current_scope(self) -> Scope:
        return self.scope_stack[-1]

    def lookup_variable(self, name: str) -> Optional[PythonType]:
        """Look up variable type in scope chain."""
        for scope in reversed(self.scope_stack):
            if name in scope.variables:
                return scope.variables[name]
        return None

    def declare_variable(self, name: str, var_type: PythonType) -> "TypedPythonState":
        """Add variable to current scope."""
        new_state = self.clone()
        new_state.current_scope.variables[name] = var_type
        return new_state


class TypeSafetyConstraint(Constraint[Dict]):
    """Constraint: expressions must be type-safe."""

    @property
    def name(self) -> str:
        return "type_safety"

    def check(self, state: SymbolicState[Dict]) -> bool:
        if not isinstance(state, TypedPythonState):
            return True

        # If we have an expected type and current type, they must match
        if state.expected_type and state.current_expr_type:
            return self._types_compatible(
                state.current_expr_type,
                state.expected_type
            )
        return True

    def _types_compatible(self, actual: PythonType, expected: PythonType) -> bool:
        """Check if types are compatible."""
        if expected == PythonType.ANY or actual == PythonType.ANY:
            return True
        if actual == expected:
            return True
        # int is compatible with float
        if expected == PythonType.FLOAT and actual == PythonType.INT:
            return True
        return False

    def to_vector(self) -> List[float]:
        return [1.0]


class TypedPythonTransition(TransitionFunction[Dict]):
    """
    Transition function for type-safe Python.

    Tracks variable declarations, type annotations, and expression types.
    """

    def __init__(self):
        # Build vocabulary
        vocab = self._build_vocabulary()

        constraints = ConstraintSet([
            TypeSafetyConstraint()
        ])

        super().__init__(vocab, constraints)

        # Type-producing operations
        self.type_producers = {
            # Literals
            "0": PythonType.INT,
            "1": PythonType.INT,
            "True": PythonType.BOOL,
            "False": PythonType.BOOL,
            "None": PythonType.NONE,
            '""': PythonType.STR,
            "[]": PythonType.LIST,
            "{}": PythonType.DICT,
            # Operations
            "+": None,  # Context-dependent
            "-": None,
            "*": None,
            "/": PythonType.FLOAT,
            "//": PythonType.INT,
            "%": PythonType.INT,
            "==": PythonType.BOOL,
            "!=": PythonType.BOOL,
            "<": PythonType.BOOL,
            ">": PythonType.BOOL,
            "and": PythonType.BOOL,
            "or": PythonType.BOOL,
            "not": PythonType.BOOL,
            "len": PythonType.INT,
            "str": PythonType.STR,
            "int": PythonType.INT,
            "float": PythonType.FLOAT,
            "bool": PythonType.BOOL,
            "list": PythonType.LIST,
        }

    def _build_vocabulary(self) -> List[str]:
        """Build Python token vocabulary."""
        vocab = [
            # Special tokens
            "<PAD>", "<BOS>", "<EOS>", "<NEWLINE>", "<INDENT>", "<DEDENT>",
            # Keywords
            "def", "return", "if", "else", "elif", "while", "for", "in",
            "class", "import", "from", "as", "with", "try", "except",
            "finally", "raise", "pass", "break", "continue", "lambda",
            "and", "or", "not", "is", "None", "True", "False",
            # Types
            "int", "float", "str", "bool", "list", "dict", "tuple", "set",
            "List", "Dict", "Tuple", "Set", "Optional", "Any", "Union",
            # Operators
            "+", "-", "*", "/", "//", "%", "**",
            "=", "+=", "-=", "*=", "/=",
            "==", "!=", "<", ">", "<=", ">=",
            "(", ")", "[", "]", "{", "}",
            ":", ",", ".", "->",
            # Literals
            "0", "1", '""', "''",
            # Variable names (simplified)
            "x", "y", "z", "i", "j", "n", "s", "v", "result", "value", "data",
            # Function names
            "len", "range", "print", "append", "pop", "get",
            # Built-ins
            "self", "cls",
        ]
        return vocab

    def transition(
        self,
        state: SymbolicState[Dict],
        token: str
    ) -> SymbolicState[Dict] | UndefinedTransition:
        """Apply transition based on token."""
        if not isinstance(state, TypedPythonState):
            return UNDEFINED

        new_state = state.clone()

        # Handle special tokens
        if token in ("<PAD>", "<BOS>", "<EOS>"):
            return new_state

        if token == "<NEWLINE>":
            new_state.in_assignment = False
            new_state.target_var = None
            new_state.expected_type = None
            new_state.current_expr_type = None
            return new_state

        if token == "<INDENT>":
            new_state.indent_level += 1
            return new_state

        if token == "<DEDENT>":
            if new_state.indent_level > 0:
                new_state.indent_level -= 1
                # Pop scope if leaving function/class
                if len(new_state.scope_stack) > 1:
                    new_state.scope_stack.pop()
            return new_state

        # Handle def (function definition)
        if token == "def":
            new_state.scope_stack.append(Scope(is_function=True))
            return new_state

        # Handle return type annotation
        if token == "->":
            return new_state

        # Handle type annotations
        if token in ("int", "float", "str", "bool", "list", "dict"):
            # If we're in assignment with target, set expected type
            if new_state.in_assignment and new_state.target_var:
                try:
                    var_type = PythonType(token)
                    new_state.expected_type = var_type
                except ValueError:
                    pass
            return new_state

        # Handle assignment
        if token == "=":
            new_state.in_assignment = True
            return new_state

        # Handle variable names (as targets or expressions)
        if token in ("x", "y", "z", "i", "j", "n", "s", "v", "result", "value", "data"):
            if new_state.in_assignment and new_state.target_var is None:
                new_state.target_var = token
            else:
                # Variable reference - look up type
                var_type = new_state.lookup_variable(token)
                if var_type:
                    new_state.current_expr_type = var_type
            return new_state

        # Handle type colon
        if token == ":":
            return new_state

        # Handle literals and type-producing operations
        if token in self.type_producers:
            produced_type = self.type_producers[token]
            if produced_type:
                new_state.current_expr_type = produced_type
            return new_state

        # Handle other tokens
        return new_state


def create_typed_python_domain() -> Tuple[TypedPythonState, TypedPythonTransition, ConstraintSet]:
    """
    Create a typed Python domain.

    Returns:
        Tuple of (initial_state, transition_function, constraints)
    """
    transition = TypedPythonTransition()
    initial_state = TypedPythonState()

    return initial_state, transition, transition.constraints


# Training data generation

def generate_typed_assignment() -> str:
    """Generate a simple typed assignment."""
    import random

    var_names = ["x", "y", "z", "result", "value"]
    types = ["int", "float", "str", "bool"]
    int_values = ["0", "1", "42", "100"]
    str_values = ['""', '"hello"', '"world"']

    var = random.choice(var_names)
    var_type = random.choice(types)

    if var_type == "int":
        value = random.choice(int_values)
    elif var_type == "str":
        value = random.choice(str_values)
    elif var_type == "bool":
        value = random.choice(["True", "False"])
    else:
        value = "0.0"

    return f"{var}: {var_type} = {value}"


def generate_simple_function() -> str:
    """Generate a simple typed function."""
    import random

    func_names = ["add", "multiply", "compute"]
    param_types = ["int", "float"]
    return_types = ["int", "float"]

    func = random.choice(func_names)
    param_type = random.choice(param_types)
    return_type = random.choice(return_types)

    return f"""def {func}(x: {param_type}, y: {param_type}) -> {return_type}:
    return x + y"""
