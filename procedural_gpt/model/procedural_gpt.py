"""
Procedural GPT: Main Model Implementation.

Dual-Path Architecture:
- Symbolic Path: σ_t updated via T (non-differentiable, provides guarantees)
- Neural Path: h_t = E_p(σ_t, R_t) (differentiable, enables learning)

Key components:
1. ProceduralEmbedding: Combines semantic, constraint, and state embeddings
2. CAB Transformer: Attention with constraint-aware biasing
3. Constraint Masking: Hard mask ensuring τ_t ∈ A(σ_{t-1})
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from procedural_gpt.core.procedural_token import ProceduralToken, SymbolicState
from procedural_gpt.core.transition import TransitionFunction
from procedural_gpt.core.constraints import ConstraintSet
from procedural_gpt.encoder.state_encoder import StateEncoder, MLPStateEncoder
from procedural_gpt.attention.cab import CABTransformerLayer
from procedural_gpt.model.embeddings import ProceduralEmbedding

S = TypeVar("S")


@dataclass
class ProceduralGPTConfig:
    """Configuration for Procedural GPT model."""

    # Model dimensions
    vocab_size: int = 32000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 2048
    max_seq_len: int = 2048

    # State encoder config
    state_dim: int = 32
    constraint_dim: int = 32
    state_hidden_dims: Optional[List[int]] = None

    # CAB config
    use_cab: bool = True
    cab_hidden_dim: int = 64
    cab_temperature: float = 1.0

    # Training config
    dropout: float = 0.1

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class ProceduralGPT(nn.Module):
    """
    Procedural GPT: Guaranteeing Generative Validity via Executable Latent States.

    This model implements the dual-path architecture:
    - Symbolic Path: State transitions via T (provides guarantees)
    - Neural Path: Embeddings via E_p (enables learning)

    Generation is conditioned on procedural tokens p_t = (σ_t, R_t),
    with hard masking ensuring only valid tokens can be generated.
    """

    def __init__(
        self,
        config: ProceduralGPTConfig,
        transition_fn: Optional[TransitionFunction] = None
    ):
        """
        Initialize Procedural GPT.

        Args:
            config: Model configuration
            transition_fn: Domain-specific transition function T
        """
        super().__init__()

        self.config = config
        self.transition_fn = transition_fn

        # Build state encoder E_p
        self.state_encoder = MLPStateEncoder(
            state_dim=config.state_dim,
            constraint_dim=config.constraint_dim,
            output_dim=config.embed_dim // 4,  # Allocate portion of embed_dim
            hidden_dims=config.state_hidden_dims or [128, 64],
            dropout=config.dropout
        )

        # Build embedding layer
        self.embedding = ProceduralEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            state_encoder=self.state_encoder,
            constraint_dim=config.constraint_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pad_token_id=config.pad_token_id
        )

        # Build transformer layers (with or without CAB)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            if config.use_cab:
                layer = CABTransformerLayer(
                    d_model=config.embed_dim,
                    nhead=config.num_heads,
                    state_dim=self.state_encoder.output_dim,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    cab_hidden_dim=config.cab_hidden_dim,
                    cab_temperature=config.cab_temperature
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=config.embed_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    batch_first=True
                )
            self.layers.append(layer)

        # Output projection
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)

        # Share weights between embedding and output
        # self.output_proj.weight = self.embedding.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        state_vectors: torch.Tensor,
        constraint_vectors: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        valid_action_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            token_ids: Input token IDs [batch, seq_len]
            state_vectors: State vectors [batch, seq_len, state_dim]
            constraint_vectors: Constraint vectors [batch, seq_len, constraint_dim]
            attention_mask: Attention mask [batch, seq_len]
            valid_action_masks: Valid action masks [batch, seq_len, vocab_size]

        Returns:
            Dictionary with logits, masked_logits, and state_embeddings
        """
        batch_size, seq_len = token_ids.shape

        # Get embeddings e_t = [e_sem; e_constr; E_p]
        hidden = self.embedding(
            token_ids,
            state_vectors,
            constraint_vectors
        )

        # Get state embeddings for CAB
        state_embeddings = self.embedding.get_state_embeddings(
            state_vectors,
            constraint_vectors
        )

        # Create causal attention mask
        causal_mask = self._create_causal_mask(seq_len, token_ids.device)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len], 1 for valid, 0 for padding
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Pass through transformer layers
        for layer in self.layers:
            if self.config.use_cab and hasattr(layer, 'self_attn'):
                # CAB layer
                hidden = layer(
                    hidden,
                    state_embeddings=state_embeddings,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask
                )
            else:
                # Standard layer
                hidden = layer(
                    hidden,
                    src_mask=causal_mask,
                    src_key_padding_mask=key_padding_mask
                )

        # Output projection
        hidden = self.output_norm(hidden)
        logits = self.output_proj(hidden)

        # Apply constraint masking if provided
        masked_logits = logits
        if valid_action_masks is not None:
            # Set invalid actions to -inf
            masked_logits = logits.masked_fill(~valid_action_masks.bool(), float('-inf'))

        return {
            'logits': logits,
            'masked_logits': masked_logits,
            'state_embeddings': state_embeddings,
            'hidden_states': hidden
        }

    def _create_causal_mask(
        self,
        size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(size, size, device=device) * float('-inf'),
            diagonal=1
        )
        return mask

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
            attention_mask: Mask for valid positions [batch, seq_len]

        Returns:
            Scalar loss value
        """
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        # Flatten
        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_targets = shift_targets.view(-1)

        # Compute loss
        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.config.pad_token_id,
            reduction='none'
        )

        # Apply mask if provided
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous().view(-1)
            loss = loss * shift_mask
            loss = loss.sum() / shift_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def set_transition_function(self, transition_fn: TransitionFunction) -> None:
        """Set or update the transition function."""
        self.transition_fn = transition_fn

    @torch.no_grad()
    def generate(
        self,
        initial_state: SymbolicState,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        constraint_set: Optional[ConstraintSet] = None
    ) -> Dict[str, Any]:
        """
        Generate sequence with constraint guarantees.

        This implements Algorithm 2 from the paper:
        Constrained Generation with Caching.

        Args:
            initial_state: Initial symbolic state σ_0
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            constraint_set: Constraint set for validation

        Returns:
            Dictionary with generated tokens, states, and metadata
        """
        if self.transition_fn is None:
            raise ValueError("Transition function required for generation")

        device = next(self.parameters()).device
        bos_token_id = bos_token_id or self.config.bos_token_id
        eos_token_id = eos_token_id or self.config.eos_token_id

        # Initialize
        current_state = initial_state.clone()
        generated_tokens = [bos_token_id]
        states = [current_state]

        # Cache for valid actions
        action_cache: Dict[int, Set[str]] = {}

        for step in range(max_length - 1):
            # Get state vector
            state_vec = torch.tensor(
                [current_state.to_vector()],
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)  # [1, 1, state_dim]

            # Pad state vector if needed
            if state_vec.shape[-1] < self.config.state_dim:
                padding = torch.zeros(
                    1, 1, self.config.state_dim - state_vec.shape[-1],
                    device=device
                )
                state_vec = torch.cat([state_vec, padding], dim=-1)

            # Get constraint vector
            constr_vec = None
            if constraint_set is not None:
                constr_list = constraint_set.to_vector()
                if len(constr_list) < self.config.constraint_dim:
                    constr_list.extend([0.0] * (self.config.constraint_dim - len(constr_list)))
                constr_vec = torch.tensor(
                    [constr_list[:self.config.constraint_dim]],
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

            # Prepare input
            token_ids = torch.tensor(
                [generated_tokens],
                dtype=torch.long,
                device=device
            )

            # Expand state vectors for full sequence
            seq_len = len(generated_tokens)
            state_vecs = state_vec.expand(1, seq_len, -1)
            constr_vecs = constr_vec.expand(1, seq_len, -1) if constr_vec is not None else None

            # Forward pass
            outputs = self.forward(
                token_ids=token_ids,
                state_vectors=state_vecs,
                constraint_vectors=constr_vecs
            )

            # Get logits for last position
            logits = outputs['logits'][0, -1, :]

            # Compute valid actions with caching
            state_hash = hash(current_state)
            if state_hash in action_cache:
                valid_actions = action_cache[state_hash]
            else:
                valid_actions = self.transition_fn.valid_actions(current_state)
                action_cache[state_hash] = valid_actions

            # Create valid action mask
            valid_mask = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=device)
            for token in valid_actions:
                idx = self.transition_fn.token_to_idx(token)
                if idx >= 0:
                    valid_mask[idx] = True

            # Always allow EOS
            valid_mask[eos_token_id] = True

            # Apply constraint mask
            logits = logits.masked_fill(~valid_mask, float('-inf'))

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Top-p filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Check for EOS
            if next_token == eos_token_id:
                break

            # Get token string and update state
            token_str = self.transition_fn.idx_to_token(next_token)
            new_state = self.transition_fn.transition(current_state, token_str)

            if new_state is None or new_state is False:
                # Should not happen with proper masking
                break

            # Update
            generated_tokens.append(next_token)
            current_state = new_state
            states.append(current_state.clone())

        return {
            'token_ids': generated_tokens,
            'tokens': [
                self.transition_fn.idx_to_token(t)
                for t in generated_tokens
                if t not in (bos_token_id, eos_token_id)
            ],
            'states': states,
            'final_state': current_state,
            'cache_hits': len(action_cache)
        }


class ProceduralGPTForTraining(ProceduralGPT):
    """
    Procedural GPT with training utilities.

    Implements Algorithm 1: Training Step (Differentiable via E_p)
    """

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        compute_auxiliary_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with teacher forcing.

        Args:
            batch: Dictionary containing:
                - token_ids: [batch, seq_len]
                - state_vectors: [batch, seq_len, state_dim]
                - constraint_vectors: [batch, seq_len, constraint_dim]
                - attention_mask: [batch, seq_len]
                - valid_action_masks: [batch, seq_len, vocab_size]
                - targets: [batch, seq_len] (optional, defaults to shifted token_ids)
            compute_auxiliary_losses: Whether to compute auxiliary losses

        Returns:
            Dictionary with loss and metrics
        """
        token_ids = batch['token_ids']
        state_vectors = batch['state_vectors']
        constraint_vectors = batch.get('constraint_vectors')
        attention_mask = batch.get('attention_mask')
        valid_action_masks = batch.get('valid_action_masks')
        targets = batch.get('targets', token_ids)

        # Forward pass
        outputs = self.forward(
            token_ids=token_ids,
            state_vectors=state_vectors,
            constraint_vectors=constraint_vectors,
            attention_mask=attention_mask,
            valid_action_masks=valid_action_masks
        )

        # Use masked logits for loss computation
        logits = outputs['masked_logits']

        # Compute main loss
        loss = self.compute_loss(logits, targets, attention_mask)

        result = {
            'loss': loss,
            'logits': outputs['logits'],
            'masked_logits': outputs['masked_logits']
        }

        if compute_auxiliary_losses:
            # Auxiliary loss: encourage state embeddings to be useful
            state_emb = outputs['state_embeddings']
            state_reg = torch.mean(state_emb ** 2) * 0.01
            result['state_regularization'] = state_reg
            result['total_loss'] = loss + state_reg

        return result
