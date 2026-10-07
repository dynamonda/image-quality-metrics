"""個々の指標の実装。1指標 = 1モジュール。

新しい指標を追加するときは、このディレクトリに ``<指標名>.py`` を作り、
``image_quality_metrics/evaluate.py`` の登録表（種別に対応するもの）と
``image_quality_metrics/__init__.py`` の公開APIに追加する。
"""

from __future__ import annotations

from image_quality_metrics.metrics.psnr import psnr
from image_quality_metrics.metrics.ssim import ssim, ssim_map
from image_quality_metrics.metrics.variance_of_laplacian import variance_of_laplacian

__all__ = ["psnr", "ssim", "ssim_map", "variance_of_laplacian"]
