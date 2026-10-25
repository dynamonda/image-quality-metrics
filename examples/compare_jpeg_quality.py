"""JPEG の品質設定ごとに PSNR / SSIM を計測する（full-reference）。

実行例::

    uv run python examples/compare_jpeg_quality.py            # 合成画像
    uv run python examples/compare_jpeg_quality.py photo.png  # 手元の画像
"""

from __future__ import annotations

import io

from PIL import Image
from sample_image import load_sample_image

import image_quality_metrics as iqm

QUALITIES = (95, 75, 50, 25, 10)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    """画像を JPEG で圧縮・展開し、劣化後の画像を返す。"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    """品質を下げながら評価し、表形式で出力する。"""
    original = load_sample_image()

    print(f"{'quality':>7}  {'psnr [dB]':>9}  {'ssim':>6}")
    for quality in QUALITIES:
        compressed = jpeg_roundtrip(original, quality)
        # PIL.Image をそのまま渡せる。evaluate は全 full-reference 指標を返す。
        scores = iqm.evaluate(original, compressed)
        print(f"{quality:>7}  {scores['psnr']:>9.2f}  {scores['ssim']:>6.4f}")


if __name__ == "__main__":
    main()
