"""Export Codabench-compatible submission from a checkpoint.

Usage:
    python scripts/export_submission.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ekmr.datasets.ek100 import EK100Dataset
from ekmr.models.cr_clip import CRClip
from ekmr.utils.checkpoints import load_checkpoint
from ekmr.utils.export import build_submission
from ekmr.utils.seed import seed_everything


def main() -> None:
    """Export submission from a trained model checkpoint."""
    parser = argparse.ArgumentParser(description="Export Codabench submission")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/raw/EK100")
    parser.add_argument("--output-dir", type=str, default="submissions")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRClip()
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    dataset = EK100Dataset(root=args.data_root, split="test", num_frames=16)

    video_embeds_list: list[np.ndarray] = []
    text_embeds_list: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            video = sample["video"].unsqueeze(0).to(device)
            text_input = torch.randn(1, 768, device=device)
            v_emb = model.encode_video(video)
            t_emb = model.encode_text({"input_ids": text_input})
            video_embeds_list.append(v_emb.cpu().numpy())
            text_embeds_list.append(t_emb.cpu().numpy())

    all_video = np.concatenate(video_embeds_list, axis=0)
    all_text = np.concatenate(text_embeds_list, axis=0)

    sim_v2t = all_video @ all_text.T
    sim_t2v = all_text @ all_video.T

    zip_path = build_submission(sim_v2t, sim_t2v, args.output_dir)
    print(f"Submission saved to: {zip_path}")
    print(f"  V2T shape: {sim_v2t.shape}")
    print(f"  T2V shape: {sim_t2v.shape}")


if __name__ == "__main__":
    main()
