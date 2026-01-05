"""
Embedding Layer for Procedural GPT.

The complete embedding at step t is:
e_t = [e_sem(τ_t); e_constr(R_t); E_p(σ_t, R_t)]

Where:
- e_sem(τ_t): Semantic token embedding
- e_constr(R_t): Constraint embedding
- E_p(σ_t, R_t): Procedural state embedding from neural encoder
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn

from procedural_gpt.encoder.state_encoder import StateEncoder


class ProceduralEmbedding(nn.Module):
    """
    Combined embedding layer for Procedural GPT.

    e_t = [e_sem(τ_t); e_constr(R_t); E_p(σ_t, R_t)]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        state_encoder: StateEncoder,
        constraint_dim: int = 32,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        Initialize procedural embedding.

        Args:
            vocab_size: Size of vocabulary V
            embed_dim: Total embedding dimension
            state_encoder: Neural encoder E_p
            constraint_dim: Dimension for constraint encoding
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: Padding token ID
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id

        # Calculate dimension allocation
        # Total = semantic + constraint + state
        self.state_dim = state_encoder.output_dim
        self.constraint_dim = constraint_dim
        self.semantic_dim = embed_dim - self.state_dim - constraint_dim

        assert self.semantic_dim > 0, (
            f"embed_dim ({embed_dim}) must be greater than "
            f"state_dim ({self.state_dim}) + constraint_dim ({constraint_dim})"
        )

        # Semantic token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            self.semantic_dim,
            padding_idx=pad_token_id
        )

        # Constraint embedding layer
        self.constraint_embedding = nn.Linear(constraint_dim, constraint_dim)

        # State encoder (E_p)
        self.state_encoder = state_encoder

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            embed_dim,
            dropout,
            max_seq_len
        )

        # Projection to combine all components
        self.combine_projection = nn.Linear(
            self.semantic_dim + constraint_dim + self.state_dim,
            embed_dim
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        if self.pad_token_id is not None:
            nn.init.zeros_(self.token_embedding.weight[self.pad_token_id])

    def forward(
        self,
        token_ids: torch.Tensor,
        state_vectors: torch.Tensor,
        constraint_vectors: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined embeddings.

        Args:
            token_ids: Token indices [batch, seq_len]
            state_vectors: State vectors for E_p [batch, seq_len, state_dim]
            constraint_vectors: Constraint vectors [batch, seq_len, constraint_dim]
            positions: Optional position indices [batch, seq_len]

        Returns:
            Combined embeddings [batch, seq_len, embed_dim]
        """
        batch_size, seq_len = token_ids.shape

        # 1. Semantic embedding: e_sem(τ_t)
        semantic_emb = self.token_embedding(token_ids)  # [B, L, semantic_dim]

        # 2. Constraint embedding: e_constr(R_t)
        if constraint_vectors is not None:
            constraint_emb = self.constraint_embedding(constraint_vectors)
        else:
            constraint_emb = torch.zeros(
                batch_size, seq_len, self.constraint_dim,
                device=token_ids.device, dtype=semantic_emb.dtype
            )

        # 3. State embedding: E_p(σ_t, R_t)
        # Reshape for batch processing
        state_flat = state_vectors.view(-1, state_vectors.shape[-1])
        constr_flat = None
        if constraint_vectors is not None:
            constr_flat = constraint_vectors.view(-1, constraint_vectors.shape[-1])

        state_emb_flat = self.state_encoder(state_flat, constr_flat)
        state_emb = state_emb_flat.view(batch_size, seq_len, -1)

        # 4. Concatenate all components
        combined = torch.cat([semantic_emb, constraint_emb, state_emb], dim=-1)

        # 5. Project to embed_dim
        combined = self.combine_projection(combined)

        # 6. Add positional encoding
        combined = self.pos_encoding(combined)

        # 7. Layer norm and dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        return combined

    def get_state_embeddings(
        self,
        state_vectors: torch.Tensor,
        constraint_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get only the state embeddings (for CAB).

        Args:
            state_vectors: [batch, seq_len, state_dim]
            constraint_vectors: [batch, seq_len, constraint_dim]

        Returns:
            State embeddings [batch, seq_len, state_encoder.output_dim]
        """
        batch_size, seq_len = state_vectors.shape[:2]

        state_flat = state_vectors.view(-1, state_vectors.shape[-1])
        constr_flat = None
        if constraint_vectors is not None:
            constr_flat = constraint_vectors.view(-1, constraint_vectors.shape[-1])

        state_emb_flat = self.state_encoder(state_flat, constr_flat)
        return state_emb_flat.view(batch_size, seq_len, -1)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)
        return self.dropout(x + pos_emb)
