"""
Tests for domain implementations.
"""

import pytest

from procedural_gpt.domains.balanced_parens import (
    BalancedParensState,
    BalancedParensTransition,
    create_balanced_parens_domain,
    is_balanced,
    generate_balanced_sequence,
)
from procedural_gpt.domains.sql_generation import (
    SQLState,
    SQLTransition,
    create_sql_domain,
    create_sample_schema,
    SQLClause,
)
from procedural_gpt.core.transition import UNDEFINED


class TestBalancedParens:
    """Tests for balanced parentheses domain."""

    def test_initial_state(self):
        initial, transition, constraints = create_balanced_parens_domain()
        assert initial.depth == 0
        assert constraints.check(initial)

    def test_valid_open(self):
        initial, transition, constraints = create_balanced_parens_domain(max_depth=10)

        state = initial.clone()
        new_state = transition.transition(state, "(")

        assert new_state.depth == 1
        assert constraints.check(new_state)

    def test_valid_close(self):
        initial, transition, constraints = create_balanced_parens_domain()

        state = BalancedParensState(depth=2, max_depth=10)
        new_state = transition.transition(state, ")")

        assert new_state.depth == 1
        assert constraints.check(new_state)

    def test_invalid_close_at_zero(self):
        initial, transition, constraints = create_balanced_parens_domain()

        state = BalancedParensState(depth=0, max_depth=10)
        result = transition.transition(state, ")")

        assert result is UNDEFINED

    def test_invalid_open_at_max(self):
        initial, transition, constraints = create_balanced_parens_domain(max_depth=5)

        state = BalancedParensState(depth=5, max_depth=5)
        result = transition.transition(state, "(")

        assert result is UNDEFINED

    def test_valid_actions_at_depth_zero(self):
        initial, transition, constraints = create_balanced_parens_domain(max_depth=10)

        state = BalancedParensState(depth=0, max_depth=10)
        valid = transition.valid_actions(state)

        assert "(" in valid
        # At depth 0, closing would be invalid

    def test_valid_actions_at_max_depth(self):
        initial, transition, constraints = create_balanced_parens_domain(max_depth=5)

        state = BalancedParensState(depth=5, max_depth=5)
        valid = transition.valid_actions(state)

        assert ")" in valid
        assert "(" not in valid  # Can't open at max depth

    def test_is_balanced_helper(self):
        assert is_balanced("()")
        assert is_balanced("(())")
        assert is_balanced("()()")
        assert is_balanced("")
        assert not is_balanced("(")
        assert not is_balanced(")")
        assert not is_balanced("(()")
        assert not is_balanced("())")

    def test_generate_balanced(self):
        for _ in range(10):
            seq = generate_balanced_sequence(10)
            assert is_balanced(seq)
            assert len(seq) == 10

    def test_full_sequence_generation(self):
        """Test that following valid actions always produces balanced sequences."""
        initial, transition, constraints = create_balanced_parens_domain(max_depth=10)
        import random

        for _ in range(100):
            state = initial.clone()
            sequence = []

            # Generate up to 20 tokens
            for _ in range(20):
                valid = transition.valid_actions(state)

                # Filter to just parens for simplicity
                paren_valid = [v for v in valid if v in ("(", ")")]
                if not paren_valid:
                    break

                token = random.choice(paren_valid)
                sequence.append(token)

                new_state = transition.transition(state, token)
                assert new_state is not UNDEFINED
                state = new_state

            # Close any remaining opens
            while state.depth > 0:
                state = transition.transition(state, ")")
                sequence.append(")")

            result = "".join(sequence)
            assert is_balanced(result), f"Generated invalid sequence: {result}"


class TestSQLGeneration:
    """Tests for SQL generation domain."""

    def test_initial_state(self):
        schema = create_sample_schema()
        initial, transition, constraints = create_sql_domain(schema)

        assert initial.current_clause == SQLClause.NONE
        assert len(initial.selected_columns) == 0
        assert len(initial.from_tables) == 0

    def test_select_transition(self):
        schema = create_sample_schema()
        initial, transition, constraints = create_sql_domain(schema)

        state = initial.clone()
        new_state = transition.transition(state, "SELECT")

        assert new_state.current_clause == SQLClause.SELECT

    def test_from_after_select(self):
        schema = create_sample_schema()
        initial, transition, constraints = create_sql_domain(schema)

        state = initial.clone()
        state = transition.transition(state, "SELECT")
        state = transition.transition(state, "*")
        state = transition.transition(state, "FROM")

        assert state.current_clause == SQLClause.FROM

    def test_from_before_select_invalid(self):
        schema = create_sample_schema()
        initial, transition, constraints = create_sql_domain(schema)

        state = initial.clone()
        result = transition.transition(state, "FROM")

        assert result is UNDEFINED

    def test_where_after_from(self):
        schema = create_sample_schema()
        initial, transition, constraints = create_sql_domain(schema)

        state = initial.clone()
        state = transition.transition(state, "SELECT")
        state = transition.transition(state, "*")
        state = transition.transition(state, "FROM")
        state = transition.transition(state, "users")
        state = transition.transition(state, "WHERE")

        assert state.current_clause == SQLClause.WHERE

    def test_table_added_to_from(self):
        schema = create_sample_schema()
        initial, transition, constraints = create_sql_domain(schema)

        state = initial.clone()
        state = transition.transition(state, "SELECT")
        state = transition.transition(state, "*")
        state = transition.transition(state, "FROM")
        state = transition.transition(state, "users")

        assert "users" in state.from_tables

    def test_sample_schema(self):
        schema = create_sample_schema()

        assert "users" in schema.tables
        assert "orders" in schema.tables
        assert "products" in schema.tables

        users = schema.tables["users"]
        assert "id" in users.columns
        assert "name" in users.columns
        assert "email" in users.columns

    def test_aggregate_detection(self):
        schema = create_sample_schema()
        initial, transition, constraints = create_sql_domain(schema)

        state = initial.clone()
        state = transition.transition(state, "SELECT")
        state = transition.transition(state, "COUNT")

        assert state.has_aggregate


class TestTypedPython:
    """Tests for typed Python domain."""

    def test_initial_state(self):
        from procedural_gpt.domains.typed_python import create_typed_python_domain

        initial, transition, constraints = create_typed_python_domain()

        assert initial.indent_level == 0
        assert len(initial.scope_stack) == 1
        assert not initial.in_assignment

    def test_newline_resets_context(self):
        from procedural_gpt.domains.typed_python import create_typed_python_domain

        initial, transition, constraints = create_typed_python_domain()

        state = initial.clone()
        state = transition.transition(state, "x")
        state = transition.transition(state, "=")
        assert state.in_assignment

        state = transition.transition(state, "<NEWLINE>")
        assert not state.in_assignment

    def test_indent_dedent(self):
        from procedural_gpt.domains.typed_python import create_typed_python_domain

        initial, transition, constraints = create_typed_python_domain()

        state = initial.clone()
        state = transition.transition(state, "<INDENT>")
        assert state.indent_level == 1

        state = transition.transition(state, "<INDENT>")
        assert state.indent_level == 2

        state = transition.transition(state, "<DEDENT>")
        assert state.indent_level == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
