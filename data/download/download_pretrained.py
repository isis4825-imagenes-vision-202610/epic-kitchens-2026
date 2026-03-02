"""Download pretrained model weights.

Usage:
    python data/download/download_pretrained.py
"""

from __future__ import annotations

from pathlib import Path


def download_pretrained(output_dir: str = "checkpoints/pretrained") -> None:
    """Provide instructions for downloading pretrained weights.

    Args:
        output_dir: Target directory for pretrained weights.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Pretrained Weights Download Instructions:")
    print("=" * 50)
    print()
    print("1. AVION ViT-L/14 (Ego4D+LaViLa pretrained):")
    print("   Source: https://github.com/zhaoyue-zephyrus/AVION/releases")
    print(f"   Save to: {out_path / 'avion_vitl14.pt'}")
    print()
    print("2. SMS-Loss checkpoint (EK100 fine-tuned):")
    print("   Source: https://github.com/xqwang14/SMS-Loss")
    print(f"   Save to: {out_path / 'sms_loss_ek100.pt'}")
    print()
    print("3. CR-CLIP checkpoint (2025 winner):")
    print("   Source: https://github.com/delCayr/ContextRefine-Clip")
    print(f"   Save to: {out_path / 'cr_clip_ek100.pt'}")


if __name__ == "__main__":
    download_pretrained()
