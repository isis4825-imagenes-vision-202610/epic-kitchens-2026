"""Integration test for a complete training step."""

from __future__ import annotations

import torch

from ekmr.losses.sms_loss import SMSLoss
from ekmr.models.cr_clip import CRClip


class TestTrainStep:
    """Integration tests for forward + backward pass."""

    def test_single_train_step(self):
        """A single training step should complete without errors."""
        model = CRClip(embed_dim=64, refinement_dim=32, num_heads=4, num_frames=4)
        loss_fn = SMSLoss(tau=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        video = torch.randn(4, 4, 3, 32, 32)
        text = {"input_ids": torch.randn(4, 64)}
        rel = torch.eye(4)

        # Forward
        v_emb, t_emb = model(video, text)
        loss = loss_fn(v_emb, t_emb, rel)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_two_steps_loss_finite(self):
        """Two consecutive training steps should produce finite losses."""
        model = CRClip(embed_dim=64, refinement_dim=32, num_heads=4, num_frames=4)
        loss_fn = SMSLoss(tau=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        losses = []
        for _ in range(2):
            video = torch.randn(4, 4, 3, 32, 32)
            text = {"input_ids": torch.randn(4, 64)}
            rel = torch.eye(4)

            v_emb, t_emb = model(video, text)
            loss = loss_fn(v_emb, t_emb, rel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        assert all(val == val for val in losses)  # no NaN
        assert all(val >= 0 for val in losses)
