#!/usr/bin/env python3
"""
Full Training Example for Procedural GPT.

This script demonstrates the complete training pipeline:
1. Data preparation with procedural tokens
2. Model initialization with CAB
3. Training with constraint masking
4. Evaluation with guaranteed constraint compliance
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from procedural_gpt.domains.balanced_parens import (
    create_balanced_parens_domain,
    create_training_data,
    is_balanced,
)
from procedural_gpt.model.procedural_gpt import ProceduralGPT, ProceduralGPTConfig
from procedural_gpt.training.data import ProceduralDataset, create_dataloader
from procedural_gpt.training.trainer import ProceduralGPTTrainer, TrainingConfig
from procedural_gpt.inference.generator import ProceduralGenerator, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Procedural GPT")

    # Data args
    parser.add_argument("--num-train", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--num-eval", type=int, default=1000, help="Number of eval samples")
    parser.add_argument("--min-length", type=int, default=4, help="Min sequence length")
    parser.add_argument("--max-length", type=int, default=30, help="Max sequence length")

    # Model args
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--use-cab", action="store_true", default=True, help="Use CAB attention")

    # Training args
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")

    return parser.parse_args()


def prepare_data(args, transition, initial_state, constraints):
    """Prepare training and evaluation datasets."""
    logger.info("Generating training data...")

    train_sequences = create_training_data(
        num_samples=args.num_train,
        min_length=args.min_length,
        max_length=args.max_length
    )
    eval_sequences = create_training_data(
        num_samples=args.num_eval,
        min_length=args.min_length,
        max_length=args.max_length
    )

    # Tokenize
    train_tokenized = [[c for c in seq] for seq in train_sequences]
    eval_tokenized = [[c for c in seq] for seq in eval_sequences]

    # Create datasets
    train_dataset = ProceduralDataset(
        sequences=train_tokenized,
        transition_fn=transition,
        initial_state=initial_state,
        constraint_set=constraints,
        max_length=args.max_length + 10,
        state_dim=4,
        constraint_dim=2
    )

    eval_dataset = ProceduralDataset(
        sequences=eval_tokenized,
        transition_fn=transition,
        initial_state=initial_state,
        constraint_set=constraints,
        max_length=args.max_length + 10,
        state_dim=4,
        constraint_dim=2
    )

    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Eval dataset: {len(eval_dataset)} examples")

    return train_dataset, eval_dataset


def create_model(args, transition):
    """Create Procedural GPT model."""
    config = ProceduralGPTConfig(
        vocab_size=transition.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.embed_dim * 4,
        max_seq_len=args.max_length + 10,
        state_dim=4,
        constraint_dim=2,
        use_cab=args.use_cab,
        dropout=0.1
    )

    model = ProceduralGPT(config, transition_fn=transition)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    return model


def evaluate_generation(model, transition, initial_state, num_samples=100):
    """Evaluate generation quality and constraint compliance."""
    generator = ProceduralGenerator(
        model=model,
        transition_fn=transition,
        device=next(model.parameters()).device
    )

    gen_config = GenerationConfig(
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_k=10
    )

    num_valid = 0
    total_length = 0
    total_cache_hits = 0

    for _ in tqdm(range(num_samples), desc="Evaluating"):
        result = generator.generate(
            initial_state=initial_state.clone(),
            config=gen_config
        )

        sequence = "".join(result.tokens)
        if is_balanced(sequence):
            num_valid += 1

        total_length += len(result.tokens)
        total_cache_hits += result.cache_hits

    validity_rate = num_valid / num_samples
    avg_length = total_length / num_samples
    avg_cache_hits = total_cache_hits / num_samples

    return {
        "validity_rate": validity_rate,
        "avg_length": avg_length,
        "avg_cache_hits": avg_cache_hits,
        "num_samples": num_samples
    }


def main():
    args = parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create domain
    logger.info("Creating balanced parentheses domain...")
    initial_state, transition, constraints = create_balanced_parens_domain(
        max_depth=15
    )

    # Prepare data
    train_dataset, eval_dataset = prepare_data(
        args, transition, initial_state, constraints
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(args, transition)

    # Create trainer
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        device=device,
        log_interval=100,
        eval_interval=500,
        save_interval=1000
    )

    trainer = ProceduralGPTTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Final evaluation
    logger.info("\nFinal Generation Evaluation:")
    model.eval()
    eval_results = evaluate_generation(
        model, transition, initial_state, num_samples=100
    )

    logger.info(f"Validity Rate: {eval_results['validity_rate']*100:.1f}%")
    logger.info(f"Avg Length: {eval_results['avg_length']:.1f}")
    logger.info(f"Avg Cache Hits: {eval_results['avg_cache_hits']:.1f}")

    # Save results
    results = {
        "args": vars(args),
        "history": history,
        "final_evaluation": eval_results
    }

    import json
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
