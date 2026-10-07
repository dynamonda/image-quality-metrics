"""入力の受け付け方（パス / PIL画像 / numpy配列）のテスト。

「どんな入力を受け付けるか」はライブラリの使い勝手を決める約束なので、
ここが壊れないようテストで固定しておく。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

import image_quality_metrics as iqm
from image_quality_metrics._core import to_array


def test_accepts_str_and_path(tmp_path: Path, gray_image: NDArray[np.uint8]) -> None:
    """ファイルパスは str でも Path でも受け付ける。"""
    path = tmp_path / "image.png"
    Image.fromarray(gray_image).save(path)

    assert iqm.psnr(path, path) == math.inf
    assert iqm.psnr(str(path), str(path)) == math.inf


def test_accepts_pil_image(gray_image: NDArray[np.uint8]) -> None:
    pil_image = Image.fromarray(gray_image)

    assert iqm.psnr(pil_image, pil_image) == math.inf


def test_mixed_input_types(tmp_path: Path, gray_image: NDArray[np.uint8]) -> None:
    """パスと配列を混ぜて渡せる（これができないと使い勝手が悪い）。"""
    path = tmp_path / "image.png"
    Image.fromarray(gray_image).save(path)

    assert iqm.psnr(path, gray_image) == math.inf


def test_png_roundtrip_is_lossless(tmp_path: Path, color_image: NDArray[np.uint8]) -> None:
    """PNG は可逆圧縮なので、保存して読み直しても完全一致する。"""
    path = tmp_path / "image.png"
    Image.fromarray(color_image).save(path)

    assert iqm.psnr(color_image, path) == math.inf


def test_rgba_is_converted_to_rgb(tmp_path: Path) -> None:
    """アルファ付き画像は RGB に変換され、3チャンネルになる。"""
    path = tmp_path / "image.png"
    Image.new("RGBA", (16, 16), (10, 20, 30, 255)).save(path)

    assert to_array(path).shape == (16, 16, 3)


def test_grayscale_keeps_single_channel(tmp_path: Path) -> None:
    """グレースケールは余計に3チャンネル化しない。"""
    path = tmp_path / "image.png"
    Image.new("L", (16, 16), 128).save(path)

    assert to_array(path).shape == (16, 16)
