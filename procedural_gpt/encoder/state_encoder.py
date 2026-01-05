"""
Neural State Encoder E_p for Procedural GPT.

E_p: S × 2^C → R^{d_p}

The encoder bridges symbolic execution with gradient-based learning:
- Maps discrete (σ_t, R_t) to continuous embeddings
- Enables gradient flow through the computational graph
- Allows transformer to learn state-space topology

Key insight: While T is non-differentiable, E_p provides a differentiable
"shadow" that enables learning of constraint-compatible behavior.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from procedural_gpt.core.procedural_token import ProceduralToken, SymbolicState
from procedural_gpt.core.constraints import ConstraintSet


class StateEncoder(ABC, nn.Module):
    """
    Abstract base class for neural state encoders E_p.

    E_p: S × 2^C → R^{d_p}

    The encoder must:
    1. Map symbolic state to a fixed-dimensional vector
    2. Encode active constraints
    3. Be differentiable for gradient flow
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(
        self,
        state_vectors: torch.Tensor,
        constraint_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode symbolic state and constraints.

        Args:
            state_vectors: Batch of state vectors [batch, state_dim]
            constraint_vectors: Batch of constraint vectors [batch, constraint_dim]

        Returns:
            Encoded representations [batch, output_dim]
        """
        pass

    def encode_procedural_token(
        self,
        token: ProceduralToken,
        constraint_set: Optional[ConstraintSet] = None
    ) -> torch.Tensor:
        """
        Encode a single procedural token.

        Args:
            token: Procedural token p_t = (σ_t, R_t)
            constraint_set: Full constraint set for encoding R_t

        Returns:
            Encoded representation [1, output_dim]
        """
        state_vec = torch.tensor(
            token.sigma.to_vector(),
            dtype=torch.float32
        ).unsqueeze(0)

        constraint_vec = None
        if constraint_set is not None:
            constraint_vec = torch.tensor(
                constraint_set.to_vector(),
                dtype=torch.float32
            ).unsqueeze(0)

        return self.forward(state_vec, constraint_vec)


class MLPStateEncoder(StateEncoder):
    """
    MLP-based state encoder.

    Architecture:
    - Input: concatenation of state vector and constraint vector
    - Hidden layers with ReLU activation
    - Output: d_p dimensional embedding

    This is the primary encoder used in Procedural GPT.
    """

    def __init__(
        self,
        state_dim: int,
        constraint_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        """
        Initialize MLP state encoder.

        Args:
            state_dim: Dimension of state vector
            constraint_dim: Dimension of constraint vector
            output_dim: Output dimension d_p
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__(output_dim)

        self.state_dim = state_dim
        self.constraint_dim = constraint_dim

        if hidden_dims is None:
            hidden_dims = [128, 64]

        input_dim = state_dim + constraint_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

        # Separate projection for state-only encoding
        self.state_proj = nn.Linear(state_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state_vectors: torch.Tensor,
        constraint_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode state and constraints.

        Args:
            state_vectors: [batch, state_dim]
            constraint_vectors: [batch, constraint_dim] or None

        Returns:
            [batch, output_dim]
        """
        if constraint_vectors is not None:
            # Full encoding with constraints
            combined = torch.cat([state_vectors, constraint_vectors], dim=-1)
            return self.encoder(combined)
        else:
            # State-only encoding
            return self.state_proj(state_vectors)


class TransformerStateEncoder(StateEncoder):
    """
    Transformer-based state encoder for complex state representations.

    Useful when state has variable-length or structured components.
    """

    def __init__(
        self,
        state_dim: int,
        constraint_dim: int,
        output_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__(output_dim)

        self.state_dim = state_dim
        self.constraint_dim = constraint_dim
        self.d_model = output_dim

        # Input projections
        self.state_embed = nn.Linear(state_dim, output_dim)
        self.constraint_embed = nn.Linear(constraint_dim, output_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(output_dim, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        state_vectors: torch.Tensor,
        constraint_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = state_vectors.shape[0]

        # Embed state
        state_emb = self.state_embed(state_vectors).unsqueeze(1)  # [B, 1, d]

        if constraint_vectors is not None:
            # Embed constraints
            constr_emb = self.constraint_embed(constraint_vectors).unsqueeze(1)
            # Concatenate as sequence
            seq = torch.cat([state_emb, constr_emb], dim=1)  # [B, 2, d]
        else:
            seq = state_emb  # [B, 1, d]

        # Apply positional encoding and transformer
        seq = self.pos_encoding(seq)
        encoded = self.transformer(seq)

        # Take first position (CLS-style) as output
        output = self.output_proj(encoded[:, 0, :])

        return output


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StateEncoderWithMemory(StateEncoder):
    """
    State encoder with recurrent memory for tracking state history.

    Useful for domains where history matters beyond current state.
    """

    def __init__(
        self,
        state_dim: int,
        constraint_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__(output_dim)

        self.state_dim = state_dim
        self.constraint_dim = constraint_dim
        self.hidden_dim = hidden_dim

        # Input projection
        input_dim = state_dim + constraint_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GRU for memory
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Initial hidden state
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(
        self,
        state_vectors: torch.Tensor,
        constraint_vectors: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode with recurrent memory.

        Args:
            state_vectors: [batch, state_dim] or [batch, seq_len, state_dim]
            constraint_vectors: [batch, constraint_dim] or [batch, seq_len, constraint_dim]
            hidden: Previous hidden state [1, batch, hidden_dim]

        Returns:
            output: [batch, output_dim] or [batch, seq_len, output_dim]
            hidden: New hidden state [1, batch, hidden_dim]
        """
        # Handle single-step vs sequence input
        squeeze_output = False
        if state_vectors.dim() == 2:
            state_vectors = state_vectors.unsqueeze(1)
            if constraint_vectors is not None:
                constraint_vectors = constraint_vectors.unsqueeze(1)
            squeeze_output = True

        batch_size = state_vectors.shape[0]

        # Combine inputs
        if constraint_vectors is not None:
            combined = torch.cat([state_vectors, constraint_vectors], dim=-1)
        else:
            combined = F.pad(
                state_vectors,
                (0, self.constraint_dim),
                value=0
            )

        # Project input
        projected = self.input_proj(combined)

        # Initialize hidden if needed
        if hidden is None:
            hidden = self.h0.expand(1, batch_size, -1).contiguous()

        # GRU forward
        output, hidden = self.gru(projected, hidden)

        # Project output
        output = self.output_proj(output)

        if squeeze_output:
            output = output.squeeze(1)

        return output, hidden

    def reset_memory(self, batch_size: int = 1) -> torch.Tensor:
        """Get initial hidden state for given batch size."""
        return self.h0.expand(1, batch_size, -1).contiguous()
