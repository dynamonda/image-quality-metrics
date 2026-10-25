"""使用例で共通して使う入力画像の用意。

画像ファイルを同梱しなくても例を実行できるよう、引数が無いときは
エッジとグラデーションを含む合成画像を生成する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["load_sample_image"]


def _synthesize(size: int = 256) -> Image.Image:
    """縞模様・円・グラデーションを重ねた RGB 画像を作る。

    平坦な画像だと劣化してもスコアがほとんど動かないため、
    細かいエッジと滑らかな領域の両方を含めている。
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / size

    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * 12 * x * (1 + y))
    circle_radius = 0.25
    circle = ((x - 0.6) ** 2 + (y - 0.4) ** 2 < circle_radius**2).astype(np.float64)
    gradient = y

    rgb = np.stack(
        [
            0.6 * stripes + 0.4 * circle,
            0.5 * gradient + 0.5 * circle,
            0.7 * (1 - gradient) + 0.3 * stripes,
        ],
        axis=-1,
    )
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def load_sample_image() -> Image.Image:
    """コマンドライン引数の画像を読み込む。指定が無ければ合成画像を返す。"""
    if len(sys.argv) > 1:
        return Image.open(Path(sys.argv[1])).convert("RGB")
    return _synthesize()
