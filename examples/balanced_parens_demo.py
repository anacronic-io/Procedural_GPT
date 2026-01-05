#!/usr/bin/env python3
"""
Balanced Parentheses Demo for Procedural GPT.

This example demonstrates:
1. Creating a balanced parentheses domain
2. Training a model to generate balanced sequences
3. Generating with 100% constraint compliance

Example from Section 3.5 of the paper:
S = Z (nesting depth)
C(σ) = (σ ≥ 0)
T(σ, "(") = σ + 1
T(σ, ")") = σ - 1 if σ > 0, else ⊥
"""

import torch
from tqdm import tqdm

from procedural_gpt.domains.balanced_parens import (
    BalancedParensState,
    BalancedParensTransition,
    create_balanced_parens_domain,
    generate_balanced_sequence,
    is_balanced,
    create_training_data,
)
from procedural_gpt.model.procedural_gpt import ProceduralGPT, ProceduralGPTConfig
from procedural_gpt.training.data import ProceduralDataset, create_dataloader
from procedural_gpt.training.trainer import ProceduralGPTTrainer, TrainingConfig
from procedural_gpt.inference.generator import ProceduralGenerator, GenerationConfig


def demo_domain():
    """Demonstrate the balanced parentheses domain."""
    print("=" * 60)
    print("Balanced Parentheses Domain Demo")
    print("=" * 60)

    # Create domain
    initial_state, transition, constraints = create_balanced_parens_domain(
        max_depth=10
    )

    print(f"\nInitial state: depth={initial_state.depth}")
    print(f"Vocabulary size: {transition.vocab_size}")
    print(f"Vocabulary: {transition.vocabulary[:10]}...")

    # Demonstrate transitions
    print("\n--- Transition Examples ---")

    state = initial_state.clone()
    print(f"State: depth={state.depth}")

    # Open paren
    state = transition.transition(state, "(")
    print(f"After '(': depth={state.depth}")

    state = transition.transition(state, "(")
    print(f"After '(': depth={state.depth}")

    state = transition.transition(state, ")")
    print(f"After ')': depth={state.depth}")

    # Show valid actions
    print("\n--- Valid Actions ---")
    state = BalancedParensState(depth=0, max_depth=10)
    valid = transition.valid_actions(state)
    print(f"At depth 0, valid actions: {valid}")

    state = BalancedParensState(depth=1, max_depth=10)
    valid = transition.valid_actions(state)
    print(f"At depth 1, valid actions: {valid}")

    state = BalancedParensState(depth=10, max_depth=10)
    valid = transition.valid_actions(state)
    print(f"At depth 10 (max), valid actions: {valid}")

    # Generate some balanced sequences
    print("\n--- Generated Balanced Sequences ---")
    for length in [4, 8, 12]:
        seq = generate_balanced_sequence(length)
        print(f"Length {length}: {seq} (balanced: {is_balanced(seq)})")


def demo_training():
    """Demonstrate training on balanced parentheses."""
    print("\n" + "=" * 60)
    print("Training Demo")
    print("=" * 60)

    # Create domain
    initial_state, transition, constraints = create_balanced_parens_domain(
        max_depth=10
    )

    # Generate training data
    print("\nGenerating training data...")
    train_sequences = create_training_data(
        num_samples=1000,
        min_length=4,
        max_length=20
    )
    print(f"Generated {len(train_sequences)} training sequences")
    print(f"Example: {train_sequences[0]}")

    # Tokenize sequences
    tokenized = [[c for c in seq] for seq in train_sequences]

    # Create dataset
    dataset = ProceduralDataset(
        sequences=tokenized,
        transition_fn=transition,
        initial_state=initial_state,
        constraint_set=constraints,
        max_length=64,
        state_dim=4,
        constraint_dim=2
    )
    print(f"Dataset size: {len(dataset)}")

    # Create model
    config = ProceduralGPTConfig(
        vocab_size=transition.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=64,
        state_dim=4,
        constraint_dim=2,
        use_cab=True
    )

    model = ProceduralGPT(config, transition_fn=transition)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Quick training demo (just a few steps)
    print("\nRunning quick training demo (5 batches)...")

    loader = create_dataloader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for i, batch in enumerate(loader):
        if i >= 5:
            break

        outputs = model(
            token_ids=batch["token_ids"],
            state_vectors=batch["state_vectors"],
            constraint_vectors=batch.get("constraint_vectors"),
            valid_action_masks=batch.get("valid_action_masks")
        )

        loss = model.compute_loss(
            outputs["masked_logits"],
            batch["token_ids"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Batch {i+1}: loss={loss.item():.4f}")

    print("Training demo complete!")
    return model, transition, initial_state


def demo_generation(model, transition, initial_state):
    """Demonstrate constrained generation."""
    print("\n" + "=" * 60)
    print("Generation Demo")
    print("=" * 60)

    # Create generator
    generator = ProceduralGenerator(
        model=model,
        transition_fn=transition,
        device="cpu"
    )

    # Generate sequences
    print("\nGenerating sequences with constraint guarantees...")

    gen_config = GenerationConfig(
        max_length=30,
        do_sample=True,
        temperature=0.8,
        top_k=10
    )

    num_valid = 0
    num_generated = 10

    for i in range(num_generated):
        result = generator.generate(
            initial_state=initial_state.clone(),
            config=gen_config
        )

        sequence = "".join(result.tokens)
        valid = is_balanced(sequence)
        num_valid += int(valid)

        print(f"  {i+1}. {sequence[:50]} (valid: {valid}, steps: {result.num_steps})")

    print(f"\nValidity rate: {num_valid}/{num_generated} ({100*num_valid/num_generated:.1f}%)")
    print("Note: With properly trained model, this should be 100%")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("PROCEDURAL GPT - BALANCED PARENTHESES DEMO")
    print("=" * 60)

    # Domain demo
    demo_domain()

    # Training demo
    model, transition, initial_state = demo_training()

    # Generation demo
    demo_generation(model, transition, initial_state)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
