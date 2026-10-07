"""個々の指標の実装。1指標 = 1モジュール。

新しい指標を追加するときは、このディレクトリに ``<指標名>.py`` を作り、
``image_quality_metrics/__init__.py`` の公開APIに追加する。
"""

from __future__ import annotations

from image_quality_metrics.metrics.psnr import psnr

__all__ = ["psnr"]
