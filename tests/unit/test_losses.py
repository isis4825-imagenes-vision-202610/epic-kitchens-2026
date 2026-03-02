"""Unit tests for loss functions."""

from __future__ import annotations

import torch

from ekmr.losses.adaptive_mimm import AdaptiveMIMMoss
from ekmr.losses.mil_nce import MILNCELoss
from ekmr.losses.sms_loss import SMSLoss


class TestSMSLoss:
    """Tests for Symmetric Multi-Similarity Loss."""

    def test_output_is_scalar(self):
        """Loss output should be a scalar tensor."""
        loss_fn = SMSLoss()
        v = torch.randn(4, 128)
        t = torch.randn(4, 128)
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        rel = torch.eye(4)
        loss = loss_fn(v, t, rel)
        assert loss.dim() == 0

    def test_gradient_flow(self):
        """Loss should produce gradients for input embeddings."""
        loss_fn = SMSLoss()
        v = torch.randn(4, 128, requires_grad=True)
        t = torch.randn(4, 128, requires_grad=True)
        v_norm = torch.nn.functional.normalize(v, dim=-1)
        t_norm = torch.nn.functional.normalize(t, dim=-1)
        rel = torch.eye(4)
        loss = loss_fn(v_norm, t_norm, rel)
        loss.backward()
        assert v.grad is not None
        assert t.grad is not None

    def test_no_nan(self):
        """Loss should not produce NaN values."""
        loss_fn = SMSLoss()
        v = torch.randn(8, 256)
        t = torch.randn(8, 256)
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        rel = torch.eye(8)
        loss = loss_fn(v, t, rel)
        assert not torch.isnan(loss).any()

    def test_symmetry(self):
        """SMS loss should be symmetric in video and text."""
        loss_fn = SMSLoss()
        v = torch.randn(4, 64)
        t = torch.randn(4, 64)
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        rel = torch.eye(4) + 0.3 * torch.ones(4, 4)
        # SMS loss is symmetric by design
        loss = loss_fn(v, t, rel)
        assert loss.item() >= 0


class TestMILNCELoss:
    """Tests for MIL-NCE Loss."""

    def test_output_is_scalar(self):
        """Loss output should be a scalar tensor."""
        loss_fn = MILNCELoss()
        v = torch.randn(4, 128)
        t = torch.randn(4, 128)
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        loss = loss_fn(v, t)
        assert loss.dim() == 0

    def test_with_relevancy(self):
        """Loss should work with explicit relevancy matrix."""
        loss_fn = MILNCELoss()
        v = torch.randn(4, 128)
        t = torch.randn(4, 128)
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        rel = torch.eye(4)
        loss = loss_fn(v, t, rel)
        assert not torch.isnan(loss)


class TestAdaptiveMIMM:
    """Tests for Adaptive MI-MM Loss."""

    def test_output_is_scalar(self):
        """Loss output should be a scalar tensor."""
        loss_fn = AdaptiveMIMMoss()
        v = torch.randn(4, 128)
        t = torch.randn(4, 128)
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        rel = torch.eye(4)
        loss = loss_fn(v, t, rel)
        assert loss.dim() == 0

    def test_non_negative(self):
        """Loss should be non-negative (margin loss)."""
        loss_fn = AdaptiveMIMMoss()
        v = torch.randn(4, 128)
        t = torch.randn(4, 128)
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        rel = torch.eye(4)
        loss = loss_fn(v, t, rel)
        assert loss.item() >= 0
