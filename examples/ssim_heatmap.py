"""SSIM マップを画像として保存し、劣化箇所を可視化する。

画像の一部だけを劣化させ、その領域が暗く写ることを確認する。
出力は ``examples/output/`` に保存する（git 管理外）。

実行例::

    uv run python examples/ssim_heatmap.py            # 合成画像
    uv run python examples/ssim_heatmap.py photo.png  # 手元の画像
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sample_image import load_sample_image

import image_quality_metrics as iqm

OUTPUT_DIR = Path(__file__).parent / "output"


def degrade_center(image: Image.Image) -> Image.Image:
    """画像の中央 1/4 の領域だけをぼかす。"""
    width, height = image.size
    box = (width // 4, height // 4, width * 3 // 4, height * 3 // 4)
    degraded = image.copy()
    degraded.paste(image.crop(box).filter(ImageFilter.GaussianBlur(3)), box)
    return degraded


def main() -> None:
    """SSIM マップをグレースケール画像に変換して保存する。"""
    original = load_sample_image()
    degraded = degrade_center(original)

    # (H-10, W-10, C)。チャンネル平均を取って1枚のマップにする。
    similarity = iqm.ssim_map(original, degraded).mean(axis=-1)

    # SSIM は [-1, 1] だが実用上は [0, 1] に収まるため、そこを 0-255 に割り当てる。
    # 白いほど元画像に近く、黒いほど劣化している。
    heatmap = (np.clip(similarity, 0, 1) * 255).astype(np.uint8)

    OUTPUT_DIR.mkdir(exist_ok=True)
    degraded.save(OUTPUT_DIR / "degraded.png")
    Image.fromarray(heatmap).save(OUTPUT_DIR / "ssim_heatmap.png")

    print(f"mean SSIM: {similarity.mean():.4f}")
    print(f"保存先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
