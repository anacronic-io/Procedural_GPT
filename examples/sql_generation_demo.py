#!/usr/bin/env python3
"""
SQL Generation Demo for Procedural GPT.

This example demonstrates:
1. Creating a SQL domain with schema constraints
2. Generating valid SQL queries
3. Guaranteed clause ordering and column validity
"""

import torch

from procedural_gpt.domains.sql_generation import (
    SQLState,
    SQLTransition,
    create_sql_domain,
    create_sample_schema,
    generate_simple_query,
    generate_query_with_where,
)
from procedural_gpt.model.procedural_gpt import ProceduralGPT, ProceduralGPTConfig
from procedural_gpt.inference.generator import ProceduralGenerator, GenerationConfig


def demo_sql_domain():
    """Demonstrate the SQL domain."""
    print("=" * 60)
    print("SQL Generation Domain Demo")
    print("=" * 60)

    # Create schema
    schema = create_sample_schema()

    print("\n--- Database Schema ---")
    for table_name, table in schema.tables.items():
        print(f"\n{table_name}:")
        for col_name, col in table.columns.items():
            pk = " [PK]" if col.is_primary_key else ""
            fk = f" [FK -> {col.references}]" if col.is_foreign_key else ""
            print(f"  - {col_name}: {col.col_type.value}{pk}{fk}")

    # Create domain
    initial_state, transition, constraints = create_sql_domain(schema)

    print(f"\nVocabulary size: {transition.vocab_size}")
    print(f"Sample vocabulary: {transition.vocabulary[:20]}...")

    # Demonstrate valid transitions
    print("\n--- Valid Transitions ---")

    state = initial_state.clone()
    print(f"Initial clause: {state.current_clause}")

    # Valid: SELECT
    print("\nAfter 'SELECT':")
    state = transition.transition(state, "SELECT")
    print(f"  Clause: {state.current_clause}")

    # Valid: *
    state = transition.transition(state, "*")
    print(f"  Selected columns: {state.selected_columns}")

    # Valid: FROM
    print("\nAfter 'FROM':")
    state = transition.transition(state, "FROM")
    print(f"  Clause: {state.current_clause}")

    # Valid: users
    state = transition.transition(state, "users")
    print(f"  From tables: {state.from_tables}")

    # Show invalid transition
    print("\n--- Invalid Transition Example ---")
    fresh_state = initial_state.clone()
    result = transition.transition(fresh_state, "FROM")  # FROM before SELECT
    print(f"FROM before SELECT: {'undefined (blocked)' if result is False or result is None else 'allowed'}")


def demo_sql_generation():
    """Demonstrate SQL query generation."""
    print("\n" + "=" * 60)
    print("SQL Generation Demo")
    print("=" * 60)

    schema = create_sample_schema()
    initial_state, transition, constraints = create_sql_domain(schema)

    # Create model
    config = ProceduralGPTConfig(
        vocab_size=transition.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=128,
        state_dim=16,
        constraint_dim=4,
        use_cab=True
    )

    model = ProceduralGPT(config, transition_fn=transition)
    model.eval()

    # Create generator
    generator = ProceduralGenerator(
        model=model,
        transition_fn=transition,
        constraint_set=constraints,
        device="cpu"
    )

    print("\n--- Generated Queries (untrained model, constrained) ---")

    gen_config = GenerationConfig(
        max_length=50,
        do_sample=True,
        temperature=1.0,
        top_k=20
    )

    for i in range(5):
        result = generator.generate(
            initial_state=initial_state.clone(),
            config=gen_config
        )

        query = " ".join(result.tokens)
        print(f"\n{i+1}. {query}")
        print(f"   Steps: {result.num_steps}, Cache hits: {result.cache_hits}")

    # Show sample valid queries
    print("\n--- Sample Valid Queries from Schema ---")
    for i in range(5):
        query = generate_simple_query(schema)
        print(f"  {query}")


def main():
    """Run SQL demos."""
    print("\n" + "=" * 60)
    print("PROCEDURAL GPT - SQL GENERATION DEMO")
    print("=" * 60)

    demo_sql_domain()
    demo_sql_generation()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
