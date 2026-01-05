"""
SQL Generation Domain.

This domain generates valid SQL queries from natural language.
State tracks:
- Schema information (tables, columns, types)
- Query structure (SELECT, FROM, WHERE, etc.)
- Join relationships
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

from procedural_gpt.core.procedural_token import SymbolicState
from procedural_gpt.core.transition import TransitionFunction, UNDEFINED, UndefinedTransition
from procedural_gpt.core.constraints import Constraint, ConstraintSet


class SQLClause(Enum):
    """SQL query clauses in order."""
    NONE = auto()
    SELECT = auto()
    FROM = auto()
    JOIN = auto()
    WHERE = auto()
    GROUP_BY = auto()
    HAVING = auto()
    ORDER_BY = auto()
    LIMIT = auto()


class ColumnType(Enum):
    """SQL column types."""
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    BOOLEAN = "BOOLEAN"


@dataclass
class Column:
    """SQL column definition."""
    name: str
    col_type: ColumnType
    table: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[Tuple[str, str]] = None  # (table, column)


@dataclass
class Table:
    """SQL table definition."""
    name: str
    columns: Dict[str, Column] = field(default_factory=dict)
    alias: Optional[str] = None


@dataclass
class Schema:
    """Database schema."""
    tables: Dict[str, Table] = field(default_factory=dict)

    def add_table(self, table: Table) -> None:
        self.tables[table.name] = table

    def get_column(self, table_name: str, column_name: str) -> Optional[Column]:
        if table_name in self.tables:
            return self.tables[table_name].columns.get(column_name)
        return None


@dataclass
class SQLState(SymbolicState[Dict]):
    """
    Symbolic state for SQL generation.

    Tracks:
    - schema: Database schema
    - current_clause: Current SQL clause being generated
    - selected_columns: Columns in SELECT
    - from_tables: Tables in FROM
    - join_tables: Tables being joined
    - where_conditions: WHERE conditions
    - group_columns: GROUP BY columns
    - order_columns: ORDER BY columns
    """

    schema: Schema = field(default_factory=Schema)
    current_clause: SQLClause = SQLClause.NONE
    selected_columns: List[str] = field(default_factory=list)
    from_tables: List[str] = field(default_factory=list)
    join_tables: List[str] = field(default_factory=list)
    where_conditions: List[str] = field(default_factory=list)
    group_columns: List[str] = field(default_factory=list)
    order_columns: List[str] = field(default_factory=list)
    has_aggregate: bool = False
    in_expression: bool = False

    def to_vector(self) -> List[float]:
        """Encode state as vector."""
        # Clause encoding (one-hot)
        clause_encoding = [0.0] * len(SQLClause)
        clause_encoding[self.current_clause.value - 1] = 1.0

        return [
            float(len(self.selected_columns)),
            float(len(self.from_tables)),
            float(len(self.join_tables)),
            float(len(self.where_conditions)),
            float(len(self.group_columns)),
            1.0 if self.has_aggregate else 0.0,
            1.0 if self.in_expression else 0.0,
        ] + clause_encoding

    def clone(self) -> "SQLState":
        return SQLState(
            schema=self.schema,  # Schema is shared (immutable)
            current_clause=self.current_clause,
            selected_columns=list(self.selected_columns),
            from_tables=list(self.from_tables),
            join_tables=list(self.join_tables),
            where_conditions=list(self.where_conditions),
            group_columns=list(self.group_columns),
            order_columns=list(self.order_columns),
            has_aggregate=self.has_aggregate,
            in_expression=self.in_expression
        )

    def __hash__(self) -> int:
        return hash((
            self.current_clause,
            tuple(self.selected_columns),
            tuple(self.from_tables),
            self.has_aggregate
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SQLState):
            return False
        return (
            self.current_clause == other.current_clause
            and self.selected_columns == other.selected_columns
            and self.from_tables == other.from_tables
        )

    def get_available_columns(self) -> List[str]:
        """Get columns available from FROM tables."""
        columns = []
        for table_name in self.from_tables + self.join_tables:
            if table_name in self.schema.tables:
                table = self.schema.tables[table_name]
                for col_name in table.columns:
                    columns.append(f"{table_name}.{col_name}")
        return columns

    def is_column_valid(self, column: str) -> bool:
        """Check if column reference is valid."""
        if "." in column:
            table, col = column.split(".", 1)
            return self.schema.get_column(table, col) is not None
        else:
            # Check all FROM tables
            for table_name in self.from_tables:
                if self.schema.get_column(table_name, column):
                    return True
        return False


class SQLClauseOrderConstraint(Constraint[Dict]):
    """Constraint: SQL clauses must appear in correct order."""

    @property
    def name(self) -> str:
        return "clause_order"

    def check(self, state: SymbolicState[Dict]) -> bool:
        # Order is enforced by transition function
        return True

    def to_vector(self) -> List[float]:
        return [1.0]


class SQLColumnValidityConstraint(Constraint[Dict]):
    """Constraint: Referenced columns must exist in schema."""

    @property
    def name(self) -> str:
        return "column_validity"

    def check(self, state: SymbolicState[Dict]) -> bool:
        if not isinstance(state, SQLState):
            return True

        # Check all selected columns are valid
        for col in state.selected_columns:
            if col != "*" and not state.is_column_valid(col):
                return False
        return True

    def to_vector(self) -> List[float]:
        return [1.0]


class SQLTransition(TransitionFunction[Dict]):
    """
    Transition function for SQL generation.

    Enforces SQL syntax and semantic constraints.
    """

    AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
    OPERATORS = {"=", "!=", "<>", "<", ">", "<=", ">=", "LIKE", "IN", "BETWEEN"}
    LOGICAL = {"AND", "OR", "NOT"}

    def __init__(self, schema: Optional[Schema] = None):
        # Build vocabulary
        vocab = self._build_vocabulary(schema)

        constraints = ConstraintSet([
            SQLClauseOrderConstraint(),
            SQLColumnValidityConstraint()
        ])

        super().__init__(vocab, constraints)
        self.schema = schema or Schema()

    def _build_vocabulary(self, schema: Optional[Schema]) -> List[str]:
        """Build SQL token vocabulary."""
        vocab = [
            # Special tokens
            "<PAD>", "<BOS>", "<EOS>",
            # Keywords
            "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
            "ON", "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "IS", "NULL",
            "ORDER", "BY", "ASC", "DESC", "GROUP", "HAVING", "LIMIT", "OFFSET",
            "AS", "DISTINCT", "ALL", "UNION", "INTERSECT", "EXCEPT",
            "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
            "CREATE", "TABLE", "DROP", "ALTER", "INDEX",
            # Aggregates
            "COUNT", "SUM", "AVG", "MIN", "MAX",
            # Operators
            "=", "!=", "<>", "<", ">", "<=", ">=",
            # Symbols
            "(", ")", ",", ".", "*", "'", '"',
            # Literals
            "0", "1", "NULL", "TRUE", "FALSE",
        ]

        # Add schema-specific tokens
        if schema:
            for table_name, table in schema.tables.items():
                vocab.append(table_name)
                for col_name in table.columns:
                    if col_name not in vocab:
                        vocab.append(col_name)

        return vocab

    def transition(
        self,
        state: SymbolicState[Dict],
        token: str
    ) -> SymbolicState[Dict] | UndefinedTransition:
        """Apply SQL transition."""
        if not isinstance(state, SQLState):
            return UNDEFINED

        new_state = state.clone()
        token_upper = token.upper()

        # Handle special tokens
        if token in ("<PAD>", "<BOS>", "<EOS>"):
            return new_state

        # Clause transitions
        if token_upper == "SELECT":
            if new_state.current_clause != SQLClause.NONE:
                return UNDEFINED  # SELECT must be first
            new_state.current_clause = SQLClause.SELECT
            return new_state

        if token_upper == "FROM":
            if new_state.current_clause != SQLClause.SELECT:
                return UNDEFINED  # FROM must follow SELECT
            if not new_state.selected_columns:
                return UNDEFINED  # Must have selected something
            new_state.current_clause = SQLClause.FROM
            return new_state

        if token_upper in ("JOIN", "LEFT", "RIGHT", "INNER", "OUTER"):
            if new_state.current_clause not in (SQLClause.FROM, SQLClause.JOIN):
                return UNDEFINED
            new_state.current_clause = SQLClause.JOIN
            return new_state

        if token_upper == "WHERE":
            if new_state.current_clause not in (SQLClause.FROM, SQLClause.JOIN):
                return UNDEFINED
            new_state.current_clause = SQLClause.WHERE
            return new_state

        if token_upper == "GROUP":
            if new_state.current_clause not in (SQLClause.FROM, SQLClause.JOIN, SQLClause.WHERE):
                return UNDEFINED
            new_state.current_clause = SQLClause.GROUP_BY
            return new_state

        if token_upper == "ORDER":
            if new_state.current_clause in (SQLClause.NONE, SQLClause.SELECT):
                return UNDEFINED
            new_state.current_clause = SQLClause.ORDER_BY
            return new_state

        if token_upper == "LIMIT":
            new_state.current_clause = SQLClause.LIMIT
            return new_state

        # Handle aggregates
        if token_upper in self.AGGREGATES:
            new_state.has_aggregate = True
            return new_state

        # Handle * (select all)
        if token == "*":
            if new_state.current_clause == SQLClause.SELECT:
                new_state.selected_columns.append("*")
            return new_state

        # Handle table/column names
        if new_state.current_clause == SQLClause.SELECT:
            if token not in self.AGGREGATES and token not in ("(", ")", ",", "AS"):
                if token.upper() not in ("DISTINCT", "ALL"):
                    new_state.selected_columns.append(token)

        if new_state.current_clause == SQLClause.FROM:
            if token in new_state.schema.tables or token.upper() not in (
                "AS", "JOIN", "LEFT", "RIGHT", "WHERE", "ORDER", "GROUP"
            ):
                if token in new_state.schema.tables:
                    new_state.from_tables.append(token)

        if new_state.current_clause == SQLClause.JOIN:
            if token in new_state.schema.tables:
                new_state.join_tables.append(token)

        return new_state


def create_sql_domain(schema: Optional[Schema] = None) -> Tuple[SQLState, SQLTransition, ConstraintSet]:
    """
    Create a SQL domain.

    Args:
        schema: Database schema (optional)

    Returns:
        Tuple of (initial_state, transition_function, constraints)
    """
    transition = SQLTransition(schema)
    initial_state = SQLState(schema=schema or Schema())

    return initial_state, transition, transition.constraints


def create_sample_schema() -> Schema:
    """Create a sample e-commerce database schema."""
    schema = Schema()

    # Users table
    users = Table(name="users")
    users.columns = {
        "id": Column("id", ColumnType.INTEGER, "users", is_primary_key=True),
        "name": Column("name", ColumnType.VARCHAR, "users"),
        "email": Column("email", ColumnType.VARCHAR, "users"),
        "created_at": Column("created_at", ColumnType.TIMESTAMP, "users"),
    }
    schema.add_table(users)

    # Orders table
    orders = Table(name="orders")
    orders.columns = {
        "id": Column("id", ColumnType.INTEGER, "orders", is_primary_key=True),
        "user_id": Column(
            "user_id", ColumnType.INTEGER, "orders",
            is_foreign_key=True, references=("users", "id")
        ),
        "total": Column("total", ColumnType.FLOAT, "orders"),
        "status": Column("status", ColumnType.VARCHAR, "orders"),
        "created_at": Column("created_at", ColumnType.TIMESTAMP, "orders"),
    }
    schema.add_table(orders)

    # Products table
    products = Table(name="products")
    products.columns = {
        "id": Column("id", ColumnType.INTEGER, "products", is_primary_key=True),
        "name": Column("name", ColumnType.VARCHAR, "products"),
        "price": Column("price", ColumnType.FLOAT, "products"),
        "category": Column("category", ColumnType.VARCHAR, "products"),
    }
    schema.add_table(products)

    return schema


# Training data generation

def generate_simple_query(schema: Schema) -> str:
    """Generate a simple SQL query."""
    import random

    tables = list(schema.tables.keys())
    if not tables:
        return "SELECT * FROM table"

    table = random.choice(tables)
    columns = list(schema.tables[table].columns.keys())

    if random.random() < 0.3:
        # SELECT *
        return f"SELECT * FROM {table}"
    else:
        # SELECT specific columns
        num_cols = random.randint(1, min(3, len(columns)))
        selected = random.sample(columns, num_cols)
        return f"SELECT {', '.join(selected)} FROM {table}"


def generate_query_with_where(schema: Schema) -> str:
    """Generate a query with WHERE clause."""
    import random

    base = generate_simple_query(schema)
    table = base.split("FROM ")[1].strip()

    if table in schema.tables:
        columns = list(schema.tables[table].columns.keys())
        col = random.choice(columns)

        col_info = schema.tables[table].columns[col]
        if col_info.col_type == ColumnType.INTEGER:
            condition = f"{col} > 0"
        elif col_info.col_type == ColumnType.VARCHAR:
            condition = f"{col} IS NOT NULL"
        else:
            condition = f"{col} IS NOT NULL"

        return f"{base} WHERE {condition}"

    return base
