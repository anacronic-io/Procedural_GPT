# Procedural GPT

**Guaranteeing Generative Validity via Executable Latent States**

This repository implements the Procedural GPT architecture, a neuro-symbolic approach that guarantees zero constraint violations in autoregressive generation through executable latent states.

## Overview

Large Language Models frequently violate structural constraints in formal domains like programming, logical reasoning, and symbolic planning. Procedural GPT addresses this by introducing **Procedural Tokens** — latent variables `p_t = (σ_t, R_t)` that encode symbolic state `σ_t` and active constraints `R_t`. Generation becomes a constrained traversal of a state-transition graph, where each emitted token updates the internal state via a symbolic transition function `T`.

### Key Features

- **Zero Constraint Violations by Construction**: When the transition function `T` is correctly specified, invalid sequences cannot be generated
- **Differentiable via E_p**: A neural encoder bridges symbolic execution with gradient-based learning
- **Constraint-Aware Biasing (CAB)**: An attention mechanism that accelerates convergence without affecting formal guarantees
- **Dual-Path Architecture**: Symbolic path for guarantees, neural path for learning

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Neural Path (Differentiable)                │
│  Input → Embedding [e_sem; e_constr; E_p(σ)] → Transformer+CAB  │
│                              ↓                                   │
│                         Logits f_θ                               │
│                              ↓                                   │
│                    Constraint Mask (Hard)                        │
│                              ↓                                   │
│                      Sample τ_{t+1}                              │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                  Symbolic Path (Formal Guarantees)               │
│     σ_t → Constraint Checker A(σ_t) → Update σ_{t+1}=T(σ_t,τ)  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/procedural-gpt.git
cd procedural-gpt

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### Balanced Parentheses Example

```python
from procedural_gpt.domains.balanced_parens import create_balanced_parens_domain
from procedural_gpt.model.procedural_gpt import ProceduralGPT, ProceduralGPTConfig
from procedural_gpt.inference.generator import ProceduralGenerator, GenerationConfig

# Create domain
initial_state, transition, constraints = create_balanced_parens_domain(max_depth=10)

# Create model
config = ProceduralGPTConfig(
    vocab_size=transition.vocab_size,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    use_cab=True
)
model = ProceduralGPT(config, transition_fn=transition)

# Generate with guarantees
generator = ProceduralGenerator(model, transition, constraints)
result = generator.generate(initial_state, GenerationConfig(max_length=50))

print("Generated:", "".join(result.tokens))
# Output is guaranteed to be balanced!
```

### Training

```python
from procedural_gpt.training.data import ProceduralDataset, create_dataloader
from procedural_gpt.training.trainer import ProceduralGPTTrainer, TrainingConfig

# Create dataset
dataset = ProceduralDataset(
    sequences=training_data,
    transition_fn=transition,
    initial_state=initial_state,
    constraint_set=constraints
)

# Train
trainer = ProceduralGPTTrainer(
    model=model,
    config=TrainingConfig(num_epochs=10, batch_size=32),
    train_dataset=dataset
)
history = trainer.train()
```

## Core Components

### Procedural Token

```python
# p_t = (σ_t, R_t)
from procedural_gpt.core.procedural_token import ProceduralToken, IntegerState

state = IntegerState(value=0)  # σ_t
token = ProceduralToken(
    sigma=state,
    R=frozenset(["depth_valid"]),  # Active constraints
    step=0
)
```

### Transition Function

```python
# T: S × V → S
from procedural_gpt.core.transition import TransitionFunction

class MyTransition(TransitionFunction):
    def transition(self, state, token):
        # Define state transition
        if token == "(":
            return state.with_depth(state.depth + 1)
        elif token == ")":
            if state.depth > 0:
                return state.with_depth(state.depth - 1)
            return UNDEFINED  # Invalid transition
```

### Neural Encoder E_p

```python
# E_p: S × 2^C → R^d_p
from procedural_gpt.encoder.state_encoder import MLPStateEncoder

encoder = MLPStateEncoder(
    state_dim=4,
    constraint_dim=2,
    output_dim=64,
    hidden_dims=[128, 64]
)
```

### Constraint-Aware Biasing

```python
# Attention(Q,K,V|σ) = softmax(QK^T/√d + λB(σ))V
from procedural_gpt.attention.cab import CABMultiHeadAttention

attention = CABMultiHeadAttention(
    embed_dim=256,
    num_heads=8,
    state_dim=64,
    cab_temperature=1.0
)
```

## Domains

### 1. Balanced Parentheses

Simple domain demonstrating the core concepts:
- State: nesting depth (integer)
- Constraint: depth ≥ 0
- Transitions: "(" increments, ")" decrements

### 2. Type-Safe Python

Generate Python code that passes type checking:
- State: variable bindings, types, scope stack
- Constraints: type compatibility
- Transitions: variable declarations, type annotations

### 3. SQL Generation

Generate valid SQL queries from natural language:
- State: schema, current clause, selected columns
- Constraints: clause ordering, column validity
- Transitions: SQL keywords and identifiers

## Theoretical Guarantees

**Theorem (Operational Invariance)**: Let π = (τ_1, ..., τ_n) be generated under policy P(τ_t | τ_{<t}, p_t). If:
1. T and C are correctly specified
2. Initial state is valid: C(σ_0) = true
3. Policy respects valid actions: supp(P) ⊆ A(σ_{t-1})

Then ∀t: C(σ_t) = true

**Corollary**: Under these conditions, generated sequences exhibit zero constraint violations, independent of the learned distribution P.

## Running Examples

```bash
# Balanced parentheses demo
python examples/balanced_parens_demo.py

# SQL generation demo
python examples/sql_generation_demo.py

# Full training example
python examples/full_training_example.py --num-epochs 10 --batch-size 64
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=procedural_gpt --cov-report=html
```

## Project Structure

```
procedural_gpt/
├── core/                  # Core components
│   ├── procedural_token.py    # Procedural Token definition
│   ├── transition.py          # Transition function base
│   └── constraints.py         # Constraint system
├── encoder/               # Neural encoding
│   └── state_encoder.py       # E_p implementations
├── attention/             # Attention mechanisms
│   └── cab.py                 # Constraint-Aware Biasing
├── model/                 # Model implementations
│   ├── procedural_gpt.py      # Main model
│   └── embeddings.py          # Embedding layer
├── domains/               # Domain implementations
│   ├── balanced_parens.py
│   ├── typed_python.py
│   └── sql_generation.py
├── training/              # Training utilities
│   ├── trainer.py
│   └── data.py
└── inference/             # Inference utilities
    └── generator.py
```

## Citation

```bibtex
@article{procedural_gpt,
  title={Procedural GPT: Guaranteeing Generative Validity via Executable Latent States},
  author={Anonymous},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
