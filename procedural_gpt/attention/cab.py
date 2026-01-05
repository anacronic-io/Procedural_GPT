"""
Constraint-Aware Biasing (CAB) Attention Mechanism.

CAB biases attention toward constraint-satisfying continuations:

Attention(Q, K, V | σ) = softmax(QK^T / √d + λB(σ)) V

where B(σ)_{i,j} = MLP([E_p(σ_i); E_p(σ_j)]) encodes state compatibility.

Key insight: CAB is orthogonal to hard masking.
- Hard masking: Guarantees A(σ) (correctness)
- CAB: Biases toward compatible states (efficiency)

CAB accelerates convergence without affecting formal guarantees.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintAwareBiasing(nn.Module):
    """
    Compute constraint-aware attention bias B(σ).

    B(σ)_{i,j} = MLP([E_p(σ_i); E_p(σ_j)])

    The bias encourages the model to attend to positions with
    compatible constraint states.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 1,
        temperature: float = 1.0
    ):
        """
        Initialize CAB module.

        Args:
            state_dim: Dimension of state embeddings from E_p
            hidden_dim: Hidden dimension for bias MLP
            num_heads: Number of attention heads (for head-specific biases)
            temperature: Temperature parameter λ for bias scaling
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_heads = num_heads
        self.temperature = temperature

        # MLP for computing pairwise compatibility
        # Input: concatenation of two state embeddings [E_p(σ_i); E_p(σ_j)]
        self.compatibility_mlp = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )

        # Learnable temperature per head
        self.head_temperatures = nn.Parameter(
            torch.ones(num_heads) * temperature
        )

    def forward(
        self,
        state_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention bias matrix B(σ).

        Args:
            state_embeddings: State embeddings from E_p [batch, seq_len, state_dim]
            mask: Optional attention mask [batch, seq_len, seq_len]

        Returns:
            Attention bias [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = state_embeddings.shape

        # Compute pairwise state combinations
        # state_i: [batch, seq_len, 1, state_dim]
        # state_j: [batch, 1, seq_len, state_dim]
        state_i = state_embeddings.unsqueeze(2)
        state_j = state_embeddings.unsqueeze(1)

        # Expand for pairwise combinations
        state_i = state_i.expand(-1, -1, seq_len, -1)
        state_j = state_j.expand(-1, seq_len, -1, -1)

        # Concatenate pairs: [batch, seq_len, seq_len, 2*state_dim]
        pairs = torch.cat([state_i, state_j], dim=-1)

        # Compute compatibility scores: [batch, seq_len, seq_len, num_heads]
        compatibility = self.compatibility_mlp(pairs)

        # Reshape to [batch, num_heads, seq_len, seq_len]
        bias = compatibility.permute(0, 3, 1, 2)

        # Apply per-head temperature scaling
        bias = bias * self.head_temperatures.view(1, -1, 1, 1)

        # Apply mask if provided (for causal attention)
        if mask is not None:
            if mask.dim() == 3:  # [batch, seq, seq]
                mask = mask.unsqueeze(1)  # [batch, 1, seq, seq]
            bias = bias.masked_fill(~mask.bool(), float('-inf'))

        return bias

    def compute_pairwise(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute compatibility between specific state pairs.

        Args:
            state_i: First state embedding [batch, state_dim]
            state_j: Second state embedding [batch, state_dim]

        Returns:
            Compatibility score [batch, num_heads]
        """
        pairs = torch.cat([state_i, state_j], dim=-1)
        return self.compatibility_mlp(pairs)


class CABMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Constraint-Aware Biasing.

    Attention(Q, K, V | σ) = softmax(QK^T / √d + λB(σ)) V

    This combines standard scaled dot-product attention with
    constraint-derived biases.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        state_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        cab_hidden_dim: int = 64,
        cab_temperature: float = 1.0
    ):
        """
        Initialize CAB multi-head attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            state_dim: Dimension of state embeddings for CAB
            dropout: Attention dropout probability
            bias: Whether to use bias in projections
            cab_hidden_dim: Hidden dimension for CAB MLP
            cab_temperature: Temperature for CAB biases
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Constraint-Aware Biasing module
        self.cab = ConstraintAwareBiasing(
            state_dim=state_dim,
            hidden_dim=cab_hidden_dim,
            num_heads=num_heads,
            temperature=cab_temperature
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with constraint-aware attention.

        Args:
            query: Query tensor [batch, tgt_len, embed_dim]
            key: Key tensor [batch, src_len, embed_dim]
            value: Value tensor [batch, src_len, embed_dim]
            state_embeddings: State embeddings for CAB [batch, src_len, state_dim]
            attn_mask: Attention mask [tgt_len, src_len] or [batch, tgt_len, src_len]
            key_padding_mask: Padding mask [batch, src_len]
            need_weights: Whether to return attention weights

        Returns:
            output: Attention output [batch, tgt_len, embed_dim]
            attn_weights: Attention weights if need_weights=True
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        # [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: QK^T / √d
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add CAB bias if state embeddings provided
        if state_embeddings is not None:
            # Compute B(σ): [batch, num_heads, src_len, src_len]
            # Note: For cross-attention, we use key states
            cab_bias = self.cab(state_embeddings, mask=attn_mask)

            # For self-attention (tgt_len == src_len), use directly
            # For cross-attention, we need to handle dimension mismatch
            if tgt_len == src_len:
                attn_scores = attn_scores + cab_bias
            else:
                # Interpolate or pad bias for cross-attention
                # For simplicity, use query-side states
                pass

        # Apply attention mask (causal or custom)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                attn_mask == float('-inf'),
                float('-inf')
            )

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, num_heads, tgt_len, head_dim] -> [batch, tgt_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)

        # Final projection
        output = self.out_proj(output)

        if need_weights:
            # Average attention weights across heads
            avg_weights = attn_weights.mean(dim=1)
            return output, avg_weights
        return output, None


class CABTransformerLayer(nn.Module):
    """
    Transformer layer with CAB attention.

    Replaces standard attention with CABMultiHeadAttention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        state_dim: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cab_hidden_dim: int = 64,
        cab_temperature: float = 1.0
    ):
        super().__init__()

        # CAB self-attention
        self.self_attn = CABMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            state_dim=state_dim,
            dropout=dropout,
            cab_hidden_dim=cab_hidden_dim,
            cab_temperature=cab_temperature
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            state_embeddings: State embeddings for CAB [batch, seq_len, state_dim]
            attn_mask: Attention mask
            key_padding_mask: Padding mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with CAB
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            state_embeddings=state_embeddings,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
