"""
Procedural GPT: Guaranteeing Generative Validity via Executable Latent States

This package implements the Procedural GPT architecture, which combines:
- Procedural Tokens: Latent states p_t = (σ_t, R_t) encoding symbolic state and constraints
- Neural Encoder E_p: Differentiable bridge between symbolic and neural computation
- Constraint-Aware Biasing (CAB): Attention mechanism for efficient learning
- Hard Masking: Guarantee zero constraint violations by construction
"""

from procedural_gpt.core.procedural_token import ProceduralToken, SymbolicState
from procedural_gpt.core.transition import TransitionFunction
from procedural_gpt.core.constraints import Constraint, ConstraintSet
from procedural_gpt.encoder.state_encoder import StateEncoder
from procedural_gpt.attention.cab import ConstraintAwareBiasing
from procedural_gpt.model.procedural_gpt import ProceduralGPT

__version__ = "0.1.0"

__all__ = [
    "ProceduralToken",
    "SymbolicState",
    "TransitionFunction",
    "Constraint",
    "ConstraintSet",
    "StateEncoder",
    "ConstraintAwareBiasing",
    "ProceduralGPT",
]
