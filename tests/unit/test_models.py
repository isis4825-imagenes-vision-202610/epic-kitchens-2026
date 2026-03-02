"""Unit tests for retrieval models."""

from __future__ import annotations

import torch

from ekmr.models.cr_clip import CRClip
from ekmr.models.custom import CustomRetriever
from ekmr.models.sms_loss_model import SMSLossModel
from ekmr.modules.cross_modal_refinement import CrossModalContextRefinement


class TestCRClip:
    """Tests for CR-CLIP model."""

    def test_forward_shapes(self):
        """Output embeddings should have correct shapes."""
        model = CRClip(embed_dim=128, refinement_dim=64, num_heads=4, num_frames=4)
        video = torch.randn(2, 4, 3, 32, 32)
        text = {"input_ids": torch.randn(2, 128)}
        v_emb, t_emb = model(video, text)
        assert v_emb.shape == (2, 128)
        assert t_emb.shape == (2, 128)

    def test_encode_video_normalized(self):
        """Video embeddings should be unit normalized."""
        model = CRClip(embed_dim=128, refinement_dim=64, num_heads=4, num_frames=4)
        video = torch.randn(2, 4, 3, 32, 32)
        v_emb = model.encode_video(video)
        norms = torch.norm(v_emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_text_normalized(self):
        """Text embeddings should be unit normalized."""
        model = CRClip(embed_dim=128, refinement_dim=64, num_heads=4, num_frames=4)
        text = {"input_ids": torch.randn(2, 128)}
        t_emb = model.encode_text(text)
        norms = torch.norm(t_emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestSMSLossModel:
    """Tests for SMS-Loss model."""

    def test_forward_shapes(self):
        """Output embeddings should have correct shapes."""
        model = SMSLossModel(embed_dim=128, num_frames=4)
        video = torch.randn(2, 4, 3, 32, 32)
        text = {"input_ids": torch.randn(2, 128)}
        v_emb, t_emb = model(video, text)
        assert v_emb.shape == (2, 128)
        assert t_emb.shape == (2, 128)


class TestCustomRetriever:
    """Tests for custom retriever skeleton."""

    def test_forward_shapes(self):
        """Output embeddings should have correct shapes."""
        model = CustomRetriever(embed_dim=64, num_frames=4)
        video = torch.randn(2, 4, 3, 32, 32)
        text = {"input_ids": torch.randn(2, 64)}
        v_emb, t_emb = model(video, text)
        assert v_emb.shape == (2, 64)
        assert t_emb.shape == (2, 64)


class TestCrossModalRefinement:
    """Tests for cross-modal refinement module."""

    def test_output_shapes(self):
        """Refined embeddings should have same shape as inputs."""
        module = CrossModalContextRefinement(
            embed_dim=128, refinement_dim=64, num_heads=4
        )
        v = torch.randn(4, 128)
        t = torch.randn(4, 128)
        v_out, t_out = module(v, t)
        assert v_out.shape == (4, 128)
        assert t_out.shape == (4, 128)

    def test_gradient_flow(self):
        """Gradients should flow through the refinement module."""
        module = CrossModalContextRefinement(
            embed_dim=64, refinement_dim=32, num_heads=4
        )
        v = torch.randn(2, 64, requires_grad=True)
        t = torch.randn(2, 64, requires_grad=True)
        v_out, t_out = module(v, t)
        loss = (v_out.sum() + t_out.sum())
        loss.backward()
        assert v.grad is not None
        assert t.grad is not None
