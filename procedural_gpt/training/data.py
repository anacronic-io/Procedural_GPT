"""
Data utilities for Procedural GPT training.

Handles conversion of sequences to training format with:
- Token IDs
- State vectors at each position
- Constraint vectors
- Valid action masks
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
from torch.utils.data import Dataset

from procedural_gpt.core.procedural_token import ProceduralToken, SymbolicState
from procedural_gpt.core.transition import TransitionFunction
from procedural_gpt.core.constraints import ConstraintSet

S = TypeVar("S")


@dataclass
class ProceduralExample:
    """A single training example."""
    tokens: List[str]
    token_ids: List[int]
    states: List[SymbolicState]
    state_vectors: List[List[float]]
    constraint_vectors: Optional[List[List[float]]] = None
    valid_action_masks: Optional[List[List[bool]]] = None


class ProceduralDataset(Dataset):
    """
    Dataset for Procedural GPT training.

    Each example consists of:
    - A sequence of tokens
    - The corresponding state at each position
    - Valid action masks for constrained training
    """

    def __init__(
        self,
        sequences: List[List[str]],
        transition_fn: TransitionFunction,
        initial_state: SymbolicState,
        constraint_set: Optional[ConstraintSet] = None,
        max_length: int = 512,
        pad_token: str = "<PAD>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        compute_valid_masks: bool = True,
        state_dim: int = 32,
        constraint_dim: int = 32
    ):
        """
        Initialize dataset.

        Args:
            sequences: List of token sequences
            transition_fn: Transition function T
            initial_state: Initial symbolic state σ_0
            constraint_set: Constraint set C
            max_length: Maximum sequence length
            pad_token: Padding token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            compute_valid_masks: Whether to compute valid action masks
            state_dim: Target state vector dimension
            constraint_dim: Target constraint vector dimension
        """
        self.sequences = sequences
        self.transition_fn = transition_fn
        self.initial_state = initial_state
        self.constraint_set = constraint_set
        self.max_length = max_length
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.compute_valid_masks = compute_valid_masks
        self.state_dim = state_dim
        self.constraint_dim = constraint_dim

        # Preprocess all examples
        self.examples = self._preprocess_all()

    def _preprocess_all(self) -> List[ProceduralExample]:
        """Preprocess all sequences into examples."""
        examples = []
        for seq in self.sequences:
            example = self._preprocess_sequence(seq)
            if example is not None:
                examples.append(example)
        return examples

    def _preprocess_sequence(self, tokens: List[str]) -> Optional[ProceduralExample]:
        """
        Convert a token sequence to a training example.

        This implements the symbolic path: σ_{t+1} = T(σ_t, τ_{t+1})
        """
        # Add special tokens
        full_tokens = [self.bos_token] + tokens + [self.eos_token]

        # Truncate if needed
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]

        # Convert to IDs
        token_ids = []
        for token in full_tokens:
            idx = self.transition_fn.token_to_idx(token)
            if idx < 0:
                idx = self.transition_fn.token_to_idx(self.pad_token)
            token_ids.append(idx)

        # Compute states by running transitions
        states = []
        state_vectors = []
        valid_masks = []

        current_state = self.initial_state.clone()

        for i, token in enumerate(full_tokens):
            # Store current state
            states.append(current_state.clone())

            # Get state vector
            state_vec = current_state.to_vector()
            # Pad or truncate to target dimension
            if len(state_vec) < self.state_dim:
                state_vec = state_vec + [0.0] * (self.state_dim - len(state_vec))
            else:
                state_vec = state_vec[:self.state_dim]
            state_vectors.append(state_vec)

            # Compute valid action mask
            if self.compute_valid_masks:
                mask = self.transition_fn.valid_action_mask(current_state)
                valid_masks.append(mask)

            # Apply transition (except for last token)
            if i < len(full_tokens) - 1:
                new_state = self.transition_fn.transition(current_state, token)
                if new_state is not None and new_state is not False:
                    current_state = new_state
                # If transition fails, keep current state

        # Get constraint vectors
        constraint_vectors = None
        if self.constraint_set is not None:
            base_constr_vec = self.constraint_set.to_vector()
            if len(base_constr_vec) < self.constraint_dim:
                base_constr_vec = base_constr_vec + [0.0] * (self.constraint_dim - len(base_constr_vec))
            else:
                base_constr_vec = base_constr_vec[:self.constraint_dim]
            constraint_vectors = [base_constr_vec] * len(full_tokens)

        return ProceduralExample(
            tokens=full_tokens,
            token_ids=token_ids,
            states=states,
            state_vectors=state_vectors,
            constraint_vectors=constraint_vectors,
            valid_action_masks=valid_masks if self.compute_valid_masks else None
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        result = {
            "token_ids": example.token_ids,
            "state_vectors": example.state_vectors,
        }

        if example.constraint_vectors is not None:
            result["constraint_vectors"] = example.constraint_vectors

        if example.valid_action_masks is not None:
            result["valid_action_masks"] = example.valid_action_masks

        return result


class ProceduralDataCollator:
    """
    Collate function for batching procedural examples.

    Handles padding and tensor conversion.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        state_dim: int = 32,
        constraint_dim: int = 32,
        vocab_size: int = 32000
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.state_dim = state_dim
        self.constraint_dim = constraint_dim
        self.vocab_size = vocab_size

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples."""
        batch_size = len(examples)

        # Find max length in batch
        max_len = max(len(ex["token_ids"]) for ex in examples)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        # Initialize tensors
        token_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        state_vectors = torch.zeros(batch_size, max_len, self.state_dim)

        has_constraints = "constraint_vectors" in examples[0]
        has_masks = "valid_action_masks" in examples[0]

        if has_constraints:
            constraint_vectors = torch.zeros(batch_size, max_len, self.constraint_dim)
        if has_masks:
            valid_action_masks = torch.zeros(
                batch_size, max_len, self.vocab_size,
                dtype=torch.bool
            )

        # Fill tensors
        for i, ex in enumerate(examples):
            seq_len = min(len(ex["token_ids"]), max_len)

            token_ids[i, :seq_len] = torch.tensor(ex["token_ids"][:seq_len])
            attention_mask[i, :seq_len] = 1

            for j in range(seq_len):
                state_vectors[i, j, :len(ex["state_vectors"][j])] = torch.tensor(
                    ex["state_vectors"][j]
                )

            if has_constraints:
                for j in range(seq_len):
                    constraint_vectors[i, j, :len(ex["constraint_vectors"][j])] = torch.tensor(
                        ex["constraint_vectors"][j]
                    )

            if has_masks:
                for j in range(seq_len):
                    mask = ex["valid_action_masks"][j]
                    valid_action_masks[i, j, :len(mask)] = torch.tensor(mask)

        result = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "state_vectors": state_vectors,
        }

        if has_constraints:
            result["constraint_vectors"] = constraint_vectors
        if has_masks:
            result["valid_action_masks"] = valid_action_masks

        return result


def create_dataloader(
    dataset: ProceduralDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for procedural training."""
    collator = ProceduralDataCollator(
        state_dim=dataset.state_dim,
        constraint_dim=dataset.constraint_dim,
        vocab_size=dataset.transition_fn.vocab_size
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        **kwargs
    )
