"""
Trainer for Procedural GPT.

Implements Algorithm 1: Training Step (Differentiable via E_p)

The training loop:
1. Compute state embeddings via E_p (differentiable)
2. Forward through transformer
3. Apply constraint mask
4. Compute cross-entropy loss
5. Backpropagate through E_p and transformer
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from procedural_gpt.model.procedural_gpt import ProceduralGPT, ProceduralGPTConfig
from procedural_gpt.training.data import ProceduralDataset, create_dataloader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    num_epochs: int = 10

    # Batch size
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = field(init=False)

    # Regularization
    state_regularization: float = 0.01

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps


class TrainingMetrics:
    """Track training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_ce_loss = 0.0
        self.total_reg_loss = 0.0
        self.num_steps = 0
        self.num_tokens = 0
        self.violations = 0

    def update(
        self,
        loss: float,
        ce_loss: float,
        reg_loss: float,
        num_tokens: int,
        violations: int = 0
    ):
        self.total_loss += loss
        self.total_ce_loss += ce_loss
        self.total_reg_loss += reg_loss
        self.num_steps += 1
        self.num_tokens += num_tokens
        self.violations += violations

    def get_metrics(self) -> Dict[str, float]:
        if self.num_steps == 0:
            return {}

        return {
            "loss": self.total_loss / self.num_steps,
            "ce_loss": self.total_ce_loss / self.num_steps,
            "reg_loss": self.total_reg_loss / self.num_steps,
            "tokens": self.num_tokens,
            "violations": self.violations,
            "perplexity": torch.exp(torch.tensor(self.total_ce_loss / self.num_steps)).item()
        }


class ProceduralGPTTrainer:
    """
    Trainer for Procedural GPT.

    Implements Algorithm 1 from the paper with:
    - Teacher forcing for efficient training
    - Constraint masking for hard guarantees
    - Gradient flow through E_p for differentiability
    """

    def __init__(
        self,
        model: ProceduralGPT,
        config: TrainingConfig,
        train_dataset: ProceduralDataset,
        eval_dataset: Optional[ProceduralDataset] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Procedural GPT model
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of callback functions
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks or []

        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Create data loaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        if eval_dataset:
            self.eval_loader = create_dataloader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False
            )
        else:
            self.eval_loader = None

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.learning_rate * 0.1
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and config.device == "cuda" else None

        # Metrics
        self.metrics = TrainingMetrics()
        self.global_step = 0
        self.epoch = 0

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay."""
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        return AdamW(param_groups, lr=self.config.learning_rate)

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Training history and final metrics.
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Total steps: {len(self.train_loader) * self.config.num_epochs}")

        history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": []
        }

        best_eval_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")

            # Training
            train_metrics = self._train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["learning_rate"].append(self.scheduler.get_last_lr()[0])

            logger.info(f"Train loss: {train_metrics['loss']:.4f}")
            logger.info(f"Train perplexity: {train_metrics['perplexity']:.2f}")

            # Evaluation
            if self.eval_loader is not None:
                eval_metrics = self._evaluate()
                history["eval_loss"].append(eval_metrics["loss"])

                logger.info(f"Eval loss: {eval_metrics['loss']:.4f}")
                logger.info(f"Eval perplexity: {eval_metrics['perplexity']:.2f}")

                # Save best model
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    self._save_checkpoint("best_model.pt")

            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save final model
        self._save_checkpoint("final_model.pt")

        return history

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()

        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for step, batch in enumerate(progress):
            loss = self._train_step(batch)

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Update progress bar
            progress.set_postfix({
                "loss": f"{loss:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Logging
            if self.global_step % self.config.log_interval == 0:
                metrics = self.metrics.get_metrics()
                for callback in self.callbacks:
                    callback("log", self.global_step, metrics)

            # Evaluation
            if self.eval_loader and self.global_step % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                for callback in self.callbacks:
                    callback("eval", self.global_step, eval_metrics)
                self.model.train()

            # Checkpointing
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step}.pt")

        return self.metrics.get_metrics()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform one training step.

        Implements Algorithm 1:
        1. h_t ← E_p(σ_t, R_t)  (differentiable encoding)
        2. e_t ← [e_sem(τ_t); e_constr(R_t); h_t]
        3. ℓ_t ← Transformer(e_t)
        4. Mask: ℓ_t[v] ← -∞ for v ∉ A(σ_t)
        5. L ← CrossEntropy(ℓ_t, τ_{t+1})
        6. σ_{t+1} ← T(σ_t, τ_{t+1})  (non-differentiable)
        7. Backpropagate through E_p and Transformer
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with mixed precision
        if self.scaler:
            with torch.amp.autocast('cuda'):
                outputs = self.model(
                    token_ids=batch["token_ids"],
                    state_vectors=batch["state_vectors"],
                    constraint_vectors=batch.get("constraint_vectors"),
                    attention_mask=batch.get("attention_mask"),
                    valid_action_masks=batch.get("valid_action_masks")
                )

                # Compute loss
                ce_loss = self.model.compute_loss(
                    outputs["masked_logits"],
                    batch["token_ids"],
                    batch.get("attention_mask")
                )

                # State regularization
                state_emb = outputs["state_embeddings"]
                reg_loss = torch.mean(state_emb ** 2) * self.config.state_regularization

                loss = ce_loss + reg_loss
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                token_ids=batch["token_ids"],
                state_vectors=batch["state_vectors"],
                constraint_vectors=batch.get("constraint_vectors"),
                attention_mask=batch.get("attention_mask"),
                valid_action_masks=batch.get("valid_action_masks")
            )

            ce_loss = self.model.compute_loss(
                outputs["masked_logits"],
                batch["token_ids"],
                batch.get("attention_mask")
            )

            state_emb = outputs["state_embeddings"]
            reg_loss = torch.mean(state_emb ** 2) * self.config.state_regularization

            loss = ce_loss + reg_loss
            loss = loss / self.config.gradient_accumulation_steps

            loss.backward()

        # Update metrics
        num_tokens = batch["attention_mask"].sum().item() if "attention_mask" in batch else batch["token_ids"].numel()
        self.metrics.update(
            loss.item() * self.config.gradient_accumulation_steps,
            ce_loss.item(),
            reg_loss.item(),
            num_tokens
        )

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on evaluation dataset."""
        self.model.eval()
        eval_metrics = TrainingMetrics()

        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                token_ids=batch["token_ids"],
                state_vectors=batch["state_vectors"],
                constraint_vectors=batch.get("constraint_vectors"),
                attention_mask=batch.get("attention_mask"),
                valid_action_masks=batch.get("valid_action_masks")
            )

            ce_loss = self.model.compute_loss(
                outputs["masked_logits"],
                batch["token_ids"],
                batch.get("attention_mask")
            )

            state_emb = outputs["state_embeddings"]
            reg_loss = torch.mean(state_emb ** 2) * self.config.state_regularization

            loss = ce_loss + reg_loss

            num_tokens = batch["attention_mask"].sum().item() if "attention_mask" in batch else batch["token_ids"].numel()
            eval_metrics.update(loss.item(), ce_loss.item(), reg_loss.item(), num_tokens)

        return eval_metrics.get_metrics()

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path}")
