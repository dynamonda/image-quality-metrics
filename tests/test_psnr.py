"""PSNR のテスト。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from image_quality_metrics import psnr


def test_identical_images_give_infinity(gray_image: NDArray[np.uint8]) -> None:
    """完全一致なら MSE が 0 になり、PSNR は無限大になる。"""
    assert psnr(gray_image, gray_image) == math.inf


def test_known_value() -> None:
    """全画素の差が 1、data_range が 255 なら MSE=1 で 10*log10(255^2)。"""
    reference = np.zeros((8, 8), dtype=np.uint8)
    test = np.ones((8, 8), dtype=np.uint8)

    expected = 10.0 * math.log10(255.0**2)
    assert psnr(reference, test, data_range=255) == pytest.approx(expected)


def test_worst_case_is_zero_db() -> None:
    """黒と白を比べると MSE が data_range^2 に等しくなり 0dB になる。"""
    black = np.zeros((8, 8), dtype=np.uint8)
    white = np.full((8, 8), 255, dtype=np.uint8)

    assert psnr(black, white) == pytest.approx(0.0)


def test_more_noise_lowers_score(
    gray_image: NDArray[np.uint8], rng: np.random.Generator
) -> None:
    """ノイズを強くすれば PSNR は必ず下がる（単調性）。"""
    base = gray_image.astype(np.float64)
    light = np.clip(base + rng.normal(0, 2, base.shape), 0, 255)
    heavy = np.clip(base + rng.normal(0, 20, base.shape), 0, 255)

    assert psnr(base, light, data_range=255) > psnr(base, heavy, data_range=255)


def test_scale_invariance(gray_image: NDArray[np.uint8], rng: np.random.Generator) -> None:
    """0-255 表現と 0-1 表現で同じスコアになる。"""
    reference = gray_image.astype(np.float64)
    test = np.clip(reference + rng.normal(0, 10, reference.shape), 0, 255)

    as_uint8 = psnr(reference, test, data_range=255)
    as_float = psnr(reference / 255.0, test / 255.0, data_range=1.0)
    assert as_uint8 == pytest.approx(as_float)


def test_shape_mismatch_raises() -> None:
    """full-reference 指標は形が違うと計算が定義できない。"""
    with pytest.raises(ValueError, match="同じ形"):
        psnr(np.zeros((8, 8)), np.zeros((8, 9)))


def test_color_image(color_image: NDArray[np.uint8]) -> None:
    """カラー画像も全チャンネルまとめて扱える。"""
    assert psnr(color_image, color_image) == math.inf
