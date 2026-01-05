"""
Tests for Procedural GPT model components.
"""

import pytest
import torch

from procedural_gpt.encoder.state_encoder import MLPStateEncoder, StateEncoder
from procedural_gpt.attention.cab import ConstraintAwareBiasing, CABMultiHeadAttention
from procedural_gpt.model.procedural_gpt import ProceduralGPT, ProceduralGPTConfig
from procedural_gpt.model.embeddings import ProceduralEmbedding


class TestStateEncoder:
    """Tests for neural state encoder E_p."""

    def test_mlp_encoder_forward(self):
        encoder = MLPStateEncoder(
            state_dim=4,
            constraint_dim=2,
            output_dim=32,
            hidden_dims=[16]
        )

        state_vec = torch.randn(8, 4)  # Batch of 8
        constr_vec = torch.randn(8, 2)

        output = encoder(state_vec, constr_vec)

        assert output.shape == (8, 32)

    def test_mlp_encoder_state_only(self):
        encoder = MLPStateEncoder(
            state_dim=4,
            constraint_dim=2,
            output_dim=32
        )

        state_vec = torch.randn(8, 4)
        output = encoder(state_vec, None)

        assert output.shape == (8, 32)

    def test_encoder_differentiable(self):
        """E_p must be differentiable for gradient flow."""
        encoder = MLPStateEncoder(
            state_dim=4,
            constraint_dim=2,
            output_dim=32
        )

        state_vec = torch.randn(8, 4, requires_grad=True)
        constr_vec = torch.randn(8, 2, requires_grad=True)

        output = encoder(state_vec, constr_vec)
        loss = output.sum()
        loss.backward()

        assert state_vec.grad is not None
        assert constr_vec.grad is not None


class TestCAB:
    """Tests for Constraint-Aware Biasing."""

    def test_cab_forward(self):
        cab = ConstraintAwareBiasing(
            state_dim=32,
            hidden_dim=16,
            num_heads=4
        )

        state_emb = torch.randn(2, 10, 32)  # [batch, seq, state_dim]
        bias = cab(state_emb)

        assert bias.shape == (2, 4, 10, 10)  # [batch, heads, seq, seq]

    def test_cab_pairwise(self):
        cab = ConstraintAwareBiasing(
            state_dim=32,
            hidden_dim=16,
            num_heads=4
        )

        state_i = torch.randn(2, 32)
        state_j = torch.randn(2, 32)

        compatibility = cab.compute_pairwise(state_i, state_j)
        assert compatibility.shape == (2, 4)

    def test_cab_attention(self):
        attn = CABMultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            state_dim=32,
            cab_hidden_dim=16
        )

        query = torch.randn(2, 10, 64)
        key = torch.randn(2, 10, 64)
        value = torch.randn(2, 10, 64)
        state_emb = torch.randn(2, 10, 32)

        output, _ = attn(query, key, value, state_embeddings=state_emb)
        assert output.shape == (2, 10, 64)


class TestProceduralEmbedding:
    """Tests for procedural embedding layer."""

    def test_embedding_forward(self):
        encoder = MLPStateEncoder(
            state_dim=4,
            constraint_dim=2,
            output_dim=16
        )

        embedding = ProceduralEmbedding(
            vocab_size=100,
            embed_dim=64,
            state_encoder=encoder,
            constraint_dim=2
        )

        token_ids = torch.randint(0, 100, (2, 10))
        state_vecs = torch.randn(2, 10, 4)
        constr_vecs = torch.randn(2, 10, 2)

        output = embedding(token_ids, state_vecs, constr_vecs)
        assert output.shape == (2, 10, 64)

    def test_get_state_embeddings(self):
        encoder = MLPStateEncoder(
            state_dim=4,
            constraint_dim=2,
            output_dim=16
        )

        embedding = ProceduralEmbedding(
            vocab_size=100,
            embed_dim=64,
            state_encoder=encoder,
            constraint_dim=2
        )

        state_vecs = torch.randn(2, 10, 4)
        constr_vecs = torch.randn(2, 10, 2)

        state_emb = embedding.get_state_embeddings(state_vecs, constr_vecs)
        assert state_emb.shape == (2, 10, 16)


class TestProceduralGPT:
    """Tests for main Procedural GPT model."""

    @pytest.fixture
    def model_config(self):
        return ProceduralGPTConfig(
            vocab_size=50,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            max_seq_len=32,
            state_dim=4,
            constraint_dim=2,
            use_cab=True
        )

    def test_model_forward(self, model_config):
        model = ProceduralGPT(model_config)

        token_ids = torch.randint(0, 50, (2, 10))
        state_vecs = torch.randn(2, 10, 4)
        constr_vecs = torch.randn(2, 10, 2)

        outputs = model(
            token_ids=token_ids,
            state_vectors=state_vecs,
            constraint_vectors=constr_vecs
        )

        assert "logits" in outputs
        assert "masked_logits" in outputs
        assert "state_embeddings" in outputs
        assert outputs["logits"].shape == (2, 10, 50)

    def test_model_with_masking(self, model_config):
        model = ProceduralGPT(model_config)

        token_ids = torch.randint(0, 50, (2, 10))
        state_vecs = torch.randn(2, 10, 4)

        # Create mask: only first 25 tokens valid
        valid_masks = torch.zeros(2, 10, 50, dtype=torch.bool)
        valid_masks[:, :, :25] = True

        outputs = model(
            token_ids=token_ids,
            state_vectors=state_vecs,
            valid_action_masks=valid_masks
        )

        # Masked logits should be -inf for invalid tokens
        assert torch.all(outputs["masked_logits"][:, :, 25:] == float("-inf"))

    def test_model_loss(self, model_config):
        model = ProceduralGPT(model_config)

        token_ids = torch.randint(0, 50, (2, 10))
        state_vecs = torch.randn(2, 10, 4)

        outputs = model(
            token_ids=token_ids,
            state_vectors=state_vecs
        )

        loss = model.compute_loss(outputs["logits"], token_ids)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)

    def test_model_gradient_flow(self, model_config):
        """Verify gradients flow through E_p."""
        model = ProceduralGPT(model_config)

        token_ids = torch.randint(0, 50, (2, 10))
        state_vecs = torch.randn(2, 10, 4, requires_grad=True)

        outputs = model(
            token_ids=token_ids,
            state_vectors=state_vecs
        )

        loss = model.compute_loss(outputs["logits"], token_ids)
        loss.backward()

        # Gradients should flow to state_vecs through E_p
        assert state_vecs.grad is not None
        assert not torch.all(state_vecs.grad == 0)


class TestModelWithCAB:
    """Tests for model with and without CAB."""

    def test_cab_vs_no_cab(self):
        config_cab = ProceduralGPTConfig(
            vocab_size=50,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            use_cab=True
        )

        config_no_cab = ProceduralGPTConfig(
            vocab_size=50,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            use_cab=False
        )

        model_cab = ProceduralGPT(config_cab)
        model_no_cab = ProceduralGPT(config_no_cab)

        # Both should work
        token_ids = torch.randint(0, 50, (2, 10))
        state_vecs = torch.randn(2, 10, 8)

        out_cab = model_cab(token_ids, state_vecs)
        out_no_cab = model_no_cab(token_ids, state_vecs)

        assert out_cab["logits"].shape == out_no_cab["logits"].shape

        # CAB model has more parameters
        params_cab = sum(p.numel() for p in model_cab.parameters())
        params_no_cab = sum(p.numel() for p in model_no_cab.parameters())
        assert params_cab > params_no_cab


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
