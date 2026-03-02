"""Interactive retrieval demo.

Usage:
    python scripts/demo.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ekmr.models.cr_clip import CRClip
from ekmr.utils.checkpoints import load_checkpoint
from ekmr.utils.seed import seed_everything
from ekmr.visualization.retrieval_viz import plot_top_k_retrievals


def main() -> None:
    """Run interactive retrieval demo."""
    parser = argparse.ArgumentParser(description="Retrieval demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/demo")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRClip()
    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    # Demo with synthetic data
    n_gallery = 20
    embed_dim = 768
    captions = [f"Person performs action {i} in the kitchen" for i in range(n_gallery)]

    with torch.no_grad():
        query_embed = torch.randn(1, embed_dim, device=device)
        query_embed = torch.nn.functional.normalize(query_embed, dim=-1)
        gallery_embeds = torch.randn(n_gallery, embed_dim, device=device)
        gallery_embeds = torch.nn.functional.normalize(gallery_embeds, dim=-1)

    sim_scores = (query_embed @ gallery_embeds.t()).cpu().numpy().squeeze()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_top_k_retrievals(
        query_text="Person opens a drawer",
        sim_scores=sim_scores,
        captions=captions,
        k=5,
        output_path=output_dir / "demo_retrieval.png",
    )
    print(f"Demo visualization saved to {output_dir / 'demo_retrieval.png'}")


if __name__ == "__main__":
    main()
