"""
Inference Generator for Procedural GPT.

Implements Algorithm 2: Constrained Generation with Caching

Key features:
- Hard masking guarantees A(σ_t) (zero violations)
- Valid action caching for efficiency
- Multiple sampling strategies (greedy, top-k, nucleus)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F

from procedural_gpt.core.procedural_token import ProceduralToken, SymbolicState
from procedural_gpt.core.transition import TransitionFunction
from procedural_gpt.core.constraints import ConstraintSet
from procedural_gpt.model.procedural_gpt import ProceduralGPT


@dataclass
class GenerationConfig:
    """Configuration for generation."""

    # Length
    max_length: int = 100
    min_length: int = 1

    # Sampling
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    # Beam search
    num_beams: int = 1

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Stopping
    stop_tokens: List[str] = field(default_factory=list)

    # Caching
    use_cache: bool = True


@dataclass
class GenerationResult:
    """Result of generation."""

    # Output sequence
    token_ids: List[int]
    tokens: List[str]

    # States
    states: List[SymbolicState]
    final_state: SymbolicState

    # Metadata
    num_steps: int
    cache_hits: int = 0
    cache_misses: int = 0
    violations_prevented: int = 0
    generation_time: float = 0.0


class ProceduralGenerator:
    """
    Generator for Procedural GPT with constraint guarantees.

    Implements Algorithm 2: Constrained Generation with Caching
    """

    def __init__(
        self,
        model: ProceduralGPT,
        transition_fn: TransitionFunction,
        constraint_set: Optional[ConstraintSet] = None,
        device: Optional[str] = None
    ):
        """
        Initialize generator.

        Args:
            model: Procedural GPT model
            transition_fn: Transition function T
            constraint_set: Constraint set C
            device: Device to run on
        """
        self.model = model
        self.transition_fn = transition_fn
        self.constraint_set = constraint_set
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Action cache: state_hash -> valid_actions
        self._action_cache: Dict[int, Set[str]] = {}

    @torch.no_grad()
    def generate(
        self,
        initial_state: SymbolicState,
        config: Optional[GenerationConfig] = None,
        prompt_tokens: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        Generate sequence with constraint guarantees.

        This implements Algorithm 2:
        1. Initialize cache
        2. For each step:
           a. Compute A(σ_t) with caching
           b. Forward through model
           c. Apply constraint mask
           d. Sample token
           e. Update state: σ_{t+1} = T(σ_t, τ_{t+1})
        3. Return on EOS or max_length

        Args:
            initial_state: Initial symbolic state σ_0
            config: Generation configuration
            prompt_tokens: Optional prompt tokens

        Returns:
            GenerationResult with tokens, states, and metadata
        """
        import time
        start_time = time.time()

        config = config or GenerationConfig()
        config.bos_token_id = config.bos_token_id or self.model.config.bos_token_id
        config.eos_token_id = config.eos_token_id or self.model.config.eos_token_id

        # Initialize
        current_state = initial_state.clone()
        generated_ids = [config.bos_token_id]
        states = [current_state.clone()]

        cache_hits = 0
        cache_misses = 0
        violations_prevented = 0

        # Process prompt if provided
        if prompt_tokens:
            for token in prompt_tokens:
                idx = self.transition_fn.token_to_idx(token)
                if idx >= 0:
                    generated_ids.append(idx)
                    new_state = self.transition_fn.transition(current_state, token)
                    if new_state is not None and new_state is not False:
                        current_state = new_state
                    states.append(current_state.clone())

        # Generation loop
        for step in range(len(generated_ids), config.max_length):
            # Get valid actions with caching (Algorithm 2, lines 3-6)
            state_hash = hash(current_state)
            if config.use_cache and state_hash in self._action_cache:
                valid_actions = self._action_cache[state_hash]
                cache_hits += 1
            else:
                valid_actions = self.transition_fn.valid_actions(current_state)
                if config.use_cache:
                    self._action_cache[state_hash] = valid_actions
                cache_misses += 1

            # Prepare input tensors
            token_tensor = torch.tensor(
                [generated_ids],
                dtype=torch.long,
                device=self.device
            )

            state_vecs = self._prepare_state_vectors(states)
            constr_vecs = self._prepare_constraint_vectors(len(states))

            # Forward pass (Algorithm 2, line 8)
            outputs = self.model(
                token_ids=token_tensor,
                state_vectors=state_vecs,
                constraint_vectors=constr_vecs
            )

            # Get logits for last position
            logits = outputs["logits"][0, -1, :].clone()

            # Count potential violations
            total_invalid = 0

            # Apply constraint mask (Algorithm 2, line 9)
            valid_mask = torch.zeros(
                self.model.config.vocab_size,
                dtype=torch.bool,
                device=self.device
            )

            for token in valid_actions:
                idx = self.transition_fn.token_to_idx(token)
                if idx >= 0:
                    valid_mask[idx] = True

            # Always allow EOS
            valid_mask[config.eos_token_id] = True

            # Count prevented violations
            probs_before = F.softmax(logits, dim=-1)
            invalid_prob = probs_before[~valid_mask].sum().item()
            violations_prevented += int(invalid_prob > 0.01)

            # Apply mask
            logits = logits.masked_fill(~valid_mask, float("-inf"))

            # Sample next token (Algorithm 2, line 10)
            next_token = self._sample_token(logits, config)

            # Check for EOS or stop tokens
            if next_token == config.eos_token_id:
                break

            token_str = self.transition_fn.idx_to_token(next_token)
            if token_str in config.stop_tokens:
                break

            # Update state (Algorithm 2, line 11)
            new_state = self.transition_fn.transition(current_state, token_str)
            if new_state is None or new_state is False:
                # Should not happen with proper masking
                break

            # Update tracking
            generated_ids.append(next_token)
            current_state = new_state
            states.append(current_state.clone())

        # Convert IDs to tokens
        tokens = []
        for idx in generated_ids:
            if idx not in (config.bos_token_id, config.eos_token_id, config.pad_token_id):
                try:
                    tokens.append(self.transition_fn.idx_to_token(idx))
                except IndexError:
                    pass

        generation_time = time.time() - start_time

        return GenerationResult(
            token_ids=generated_ids,
            tokens=tokens,
            states=states,
            final_state=current_state,
            num_steps=len(generated_ids) - 1,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            violations_prevented=violations_prevented,
            generation_time=generation_time
        )

    def _prepare_state_vectors(self, states: List[SymbolicState]) -> torch.Tensor:
        """Prepare state vectors tensor."""
        state_dim = self.model.config.state_dim
        seq_len = len(states)

        state_vecs = torch.zeros(1, seq_len, state_dim, device=self.device)

        for i, state in enumerate(states):
            vec = state.to_vector()
            vec_len = min(len(vec), state_dim)
            state_vecs[0, i, :vec_len] = torch.tensor(vec[:vec_len])

        return state_vecs

    def _prepare_constraint_vectors(self, seq_len: int) -> Optional[torch.Tensor]:
        """Prepare constraint vectors tensor."""
        if self.constraint_set is None:
            return None

        constr_dim = self.model.config.constraint_dim
        constr_vec = self.constraint_set.to_vector()

        if len(constr_vec) < constr_dim:
            constr_vec = constr_vec + [0.0] * (constr_dim - len(constr_vec))
        else:
            constr_vec = constr_vec[:constr_dim]

        constr_tensor = torch.tensor(
            [constr_vec],
            device=self.device
        ).unsqueeze(0).expand(1, seq_len, -1)

        return constr_tensor

    def _sample_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig
    ) -> int:
        """Sample next token based on configuration."""
        # Temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Top-k filtering
        if config.top_k is not None and config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample or greedy
        if config.do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = logits.argmax().item()

        return next_token

    def clear_cache(self) -> None:
        """Clear the action cache."""
        self._action_cache.clear()

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._action_cache)


class BeamSearchGenerator(ProceduralGenerator):
    """
    Generator with beam search support.

    Maintains multiple hypotheses while respecting constraints.
    """

    @torch.no_grad()
    def generate_beam(
        self,
        initial_state: SymbolicState,
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        Generate sequences using beam search.

        Args:
            initial_state: Initial symbolic state
            config: Generation configuration

        Returns:
            List of GenerationResult, one per beam
        """
        config = config or GenerationConfig()
        num_beams = config.num_beams

        # Initialize beams
        beams = [{
            "tokens": [config.bos_token_id],
            "state": initial_state.clone(),
            "states": [initial_state.clone()],
            "score": 0.0
        }]

        finished = []

        for step in range(config.max_length):
            all_candidates = []

            for beam in beams:
                current_state = beam["state"]

                # Get valid actions
                valid_actions = self.transition_fn.valid_actions(current_state)

                # Get model predictions
                token_tensor = torch.tensor(
                    [beam["tokens"]],
                    dtype=torch.long,
                    device=self.device
                )

                state_vecs = self._prepare_state_vectors(beam["states"])
                outputs = self.model(
                    token_ids=token_tensor,
                    state_vectors=state_vecs
                )

                logits = outputs["logits"][0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)

                # Expand beam
                for token in valid_actions:
                    idx = self.transition_fn.token_to_idx(token)
                    if idx >= 0:
                        new_score = beam["score"] + log_probs[idx].item()

                        new_state = self.transition_fn.transition(current_state, token)
                        if new_state is not None and new_state is not False:
                            candidate = {
                                "tokens": beam["tokens"] + [idx],
                                "state": new_state,
                                "states": beam["states"] + [new_state.clone()],
                                "score": new_score
                            }
                            all_candidates.append(candidate)

                # Also consider EOS
                eos_score = beam["score"] + log_probs[config.eos_token_id].item()
                finished.append({
                    "tokens": beam["tokens"],
                    "state": current_state,
                    "states": beam["states"],
                    "score": eos_score
                })

            # Select top beams
            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            beams = all_candidates[:num_beams]

            if not beams:
                break

        # Combine finished and remaining beams
        all_results = finished + beams
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Convert to GenerationResult
        results = []
        for beam in all_results[:num_beams]:
            tokens = []
            for idx in beam["tokens"]:
                if idx not in (config.bos_token_id, config.eos_token_id):
                    try:
                        tokens.append(self.transition_fn.idx_to_token(idx))
                    except IndexError:
                        pass

            results.append(GenerationResult(
                token_ids=beam["tokens"],
                tokens=tokens,
                states=beam["states"],
                final_state=beam["state"],
                num_steps=len(beam["tokens"]) - 1
            ))

        return results
