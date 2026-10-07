"""Variance of Laplacian のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from image_quality_metrics import variance_of_laplacian
from image_quality_metrics._core import filter_2d_valid, gaussian_kernel_1d

_BLUR_RADIUS = 5


def _blurred_and_sharp(
    image: NDArray[np.float64], sigma: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ボケ版と、同じ領域に切り出した元画像を返す。

    ボケは valid モードなので端が落ちる。公平に比べるため元画像も同じだけ
    切り出して形を揃える。ぼかしにライブラリ内部のガウシアンを使うことで、
    テストのために依存を増やさずに済む。
    """
    kernel = gaussian_kernel_1d(sigma=sigma, radius=_BLUR_RADIUS)
    blurred = filter_2d_valid(image, kernel)
    sharp = image[_BLUR_RADIUS:-_BLUR_RADIUS, _BLUR_RADIUS:-_BLUR_RADIUS]
    return blurred, sharp


def test_flat_image_is_zero() -> None:
    """一様な画像にはエッジがないため、応答の分散は0になる。"""
    flat = np.full((16, 16), 128, dtype=np.uint8)

    assert variance_of_laplacian(flat, data_range=255) == 0.0


def test_blur_lowers_score(gray_image: NDArray[np.uint8]) -> None:
    """ボケさせるとスコアは下がる。この指標の存在意義そのもの。"""
    blurred, sharp = _blurred_and_sharp(gray_image.astype(np.float64), sigma=2.0)

    assert variance_of_laplacian(blurred, data_range=255) < variance_of_laplacian(
        sharp, data_range=255
    )


def test_stronger_blur_lowers_score_further(gray_image: NDArray[np.uint8]) -> None:
    """ボケが強いほどスコアは低い（単調性）。"""
    base = gray_image.astype(np.float64)
    light, _ = _blurred_and_sharp(base, sigma=1.0)
    heavy, _ = _blurred_and_sharp(base, sigma=3.0)

    assert variance_of_laplacian(heavy, data_range=255) < variance_of_laplacian(
        light, data_range=255
    )


def test_scale_invariance(gray_image: NDArray[np.uint8]) -> None:
    """0-255 表現と 0-1 表現で同じスコアになる。

    分散は入力スケールの2乗で動くため、揃えておかないと表現を変えただけで
    65025 倍ずれる。
    """
    as_uint8 = variance_of_laplacian(gray_image, data_range=255)
    as_float = variance_of_laplacian(gray_image / 255.0, data_range=1.0)

    assert as_uint8 == pytest.approx(as_float)


def test_color_matches_grayscale_when_channels_are_equal(
    gray_image: NDArray[np.uint8],
) -> None:
    """R=G=B のカラー画像は、グレースケール版と同じスコアになる。

    輝度係数の総和が1であることの確認も兼ねている。
    """
    color = np.stack([gray_image] * 3, axis=-1)

    assert variance_of_laplacian(color, data_range=255) == pytest.approx(
        variance_of_laplacian(gray_image, data_range=255)
    )


def test_checkerboard_scores_higher_than_flat() -> None:
    """市松模様は全画素がエッジなので、一様な画像より高くなる。"""
    yy, xx = np.mgrid[0:32, 0:32]
    checkerboard = np.where((yy + xx) % 2 == 0, 0, 255).astype(np.float64)
    flat = np.full((32, 32), 128, dtype=np.float64)

    assert variance_of_laplacian(checkerboard, data_range=255) > variance_of_laplacian(
        flat, data_range=255
    )


def test_ignores_alpha_channel(gray_image: NDArray[np.uint8]) -> None:
    """アルファは明るさの情報ではないため、スコアに影響しない。"""
    rgb = np.stack([gray_image] * 3, axis=-1)
    rgba = np.dstack([rgb, np.full_like(gray_image, 128)])

    assert variance_of_laplacian(rgba, data_range=255) == pytest.approx(
        variance_of_laplacian(rgb, data_range=255)
    )


def test_image_smaller_than_kernel_raises() -> None:
    """3x3 のカーネルを当てられない画像は計算できない。"""
    with pytest.raises(ValueError, match="3x3 以上"):
        variance_of_laplacian(np.zeros((2, 8), dtype=np.uint8), data_range=255)


def test_accepts_path(tmp_path: Path, gray_image: NDArray[np.uint8]) -> None:
    """他の指標と同じく、パスでも渡せる。"""
    path = tmp_path / "image.png"
    Image.fromarray(gray_image).save(path)

    assert variance_of_laplacian(path, data_range=255) == pytest.approx(
        variance_of_laplacian(gray_image, data_range=255)
    )
