"""SSIM のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from image_quality_metrics import ssim, ssim_map


def test_identical_images_give_one(gray_image: NDArray[np.uint8]) -> None:
    """完全一致なら SSIM は 1。"""
    assert ssim(gray_image, gray_image) == pytest.approx(1.0)


def test_value_is_within_range(
    gray_image: NDArray[np.uint8], rng: np.random.Generator
) -> None:
    """SSIM の値域は [-1, 1]。"""
    noisy = np.clip(gray_image + rng.normal(0, 30, gray_image.shape), 0, 255)
    score = ssim(gray_image.astype(np.float64), noisy, data_range=255)

    assert -1.0 <= score <= 1.0


def test_more_noise_lowers_score(
    gray_image: NDArray[np.uint8], rng: np.random.Generator
) -> None:
    """ノイズを強くすれば SSIM は下がる（単調性）。"""
    base = gray_image.astype(np.float64)
    light = np.clip(base + rng.normal(0, 2, base.shape), 0, 255)
    heavy = np.clip(base + rng.normal(0, 30, base.shape), 0, 255)

    assert ssim(base, light, data_range=255) > ssim(base, heavy, data_range=255)


def test_scale_invariance(
    gray_image: NDArray[np.uint8], rng: np.random.Generator
) -> None:
    """0-255 表現と 0-1 表現で同じスコアになる。"""
    reference = gray_image.astype(np.float64)
    test = np.clip(reference + rng.normal(0, 10, reference.shape), 0, 255)

    as_uint8 = ssim(reference, test, data_range=255)
    as_float = ssim(reference / 255.0, test / 255.0, data_range=1.0)
    assert as_uint8 == pytest.approx(as_float)


def test_is_symmetric(gray_image: NDArray[np.uint8], rng: np.random.Generator) -> None:
    """SSIM は2枚を入れ替えても同じ値になる。"""
    base = gray_image.astype(np.float64)
    other = np.clip(base + rng.normal(0, 10, base.shape), 0, 255)

    assert ssim(base, other, data_range=255) == pytest.approx(
        ssim(other, base, data_range=255)
    )


def test_map_shape_is_reduced_by_window(gray_image: NDArray[np.uint8]) -> None:
    """端を落とす valid モードなので、マップは窓の分だけ小さくなる。"""
    result = ssim_map(gray_image, gray_image, window_size=11)

    assert result.shape == (64 - 10, 64 - 10, 1)


def test_map_shape_for_color(color_image: NDArray[np.uint8]) -> None:
    """カラー画像ではチャンネル数が保たれる。"""
    result = ssim_map(color_image, color_image, window_size=11)

    assert result.shape == (54, 54, 3)


def test_image_smaller_than_window_raises() -> None:
    """窓より小さい画像では局所統計が取れない。"""
    tiny = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="画像が窓より小さい"):
        ssim(tiny, tiny, window_size=11)


def test_even_window_size_raises(gray_image: NDArray[np.uint8]) -> None:
    """窓は中心画素を持つ必要があるため奇数でなければならない。"""
    with pytest.raises(ValueError, match="正の奇数"):
        ssim(gray_image, gray_image, window_size=10)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="同じ形"):
        ssim(np.zeros((16, 16)), np.zeros((16, 17)))
