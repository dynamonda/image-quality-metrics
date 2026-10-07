"""image-quality-metrics: 画像の品質評価指標を簡単に計測するライブラリ。

このファイルが公開APIの定義そのもの。ここで import / ``__all__`` に
載せたものが「利用者に対する約束」であり、載っていないものは内部実装として
予告なく変えてよい、という境界になっている。
"""

from __future__ import annotations

from image_quality_metrics._core import ImageLike
from image_quality_metrics.metrics.psnr import psnr
from image_quality_metrics.metrics.ssim import ssim, ssim_map

__version__ = "0.1.0"

__all__ = ["ImageLike", "__version__", "psnr", "ssim", "ssim_map"]
