"""evaluate() と公開APIのテスト。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import image_quality_metrics as iqm


def test_returns_all_metrics_by_default(gray_image: NDArray[np.uint8]) -> None:
    result = iqm.evaluate(gray_image, gray_image)

    assert set(result) == set(iqm.available_metrics())
    assert result["psnr"] == math.inf
    assert result["ssim"] == pytest.approx(1.0)


def test_preserves_requested_order(gray_image: NDArray[np.uint8]) -> None:
    """指定した順序どおりに返す（表示やCSV出力で順序が崩れないように）。"""
    result = iqm.evaluate(gray_image, gray_image, metrics=["ssim", "psnr"])

    assert list(result) == ["ssim", "psnr"]


def test_unknown_metric_raises(gray_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="未知の指標"):
        iqm.evaluate(gray_image, gray_image, metrics=["lpips"])


def test_matches_direct_calls(
    gray_image: NDArray[np.uint8], rng: np.random.Generator
) -> None:
    """evaluate() は個別関数を呼ぶだけなので、結果は一致しなければならない。"""
    base = gray_image.astype(np.float64)
    noisy = np.clip(base + rng.normal(0, 10, base.shape), 0, 255)

    result = iqm.evaluate(base, noisy, data_range=255)

    assert result["psnr"] == pytest.approx(iqm.psnr(base, noisy, data_range=255))
    assert result["ssim"] == pytest.approx(iqm.ssim(base, noisy, data_range=255))


def test_public_api_is_importable() -> None:
    """__all__ に挙げた名前が実際に存在することを確認する。"""
    for name in iqm.__all__:
        assert hasattr(iqm, name), f"{name} が公開されていません"


def test_no_reference_returns_all_metrics_by_default(
    gray_image: NDArray[np.uint8],
) -> None:
    result = iqm.evaluate_no_reference(gray_image, data_range=255)

    assert set(result) == set(iqm.available_no_reference_metrics())
    assert result["variance_of_laplacian"] > 0.0


def test_no_reference_matches_direct_call(gray_image: NDArray[np.uint8]) -> None:
    """evaluate_no_reference() は個別関数を呼ぶだけ。"""
    result = iqm.evaluate_no_reference(gray_image, data_range=255)

    assert result["variance_of_laplacian"] == pytest.approx(
        iqm.variance_of_laplacian(gray_image, data_range=255)
    )


def test_no_reference_unknown_metric_raises(gray_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="未知の指標"):
        iqm.evaluate_no_reference(gray_image, metrics=["niqe"])


def test_registries_are_disjoint() -> None:
    """同じ名前が両方の登録表にあると、どちらで呼ぶべきか曖昧になる。"""
    overlap = set(iqm.available_metrics()) & set(iqm.available_no_reference_metrics())

    assert not overlap, f"両方の登録表に存在する指標: {sorted(overlap)}"


def test_full_reference_metric_is_rejected_by_no_reference(
    gray_image: NDArray[np.uint8],
) -> None:
    """基準画像が必要な指標は evaluate_no_reference では使えない。"""
    with pytest.raises(ValueError, match="未知の指標"):
        iqm.evaluate_no_reference(gray_image, metrics=["psnr"])


def test_no_reference_metric_is_rejected_by_evaluate(
    gray_image: NDArray[np.uint8],
) -> None:
    """逆方向も同じ。登録表が分かれていることの確認。"""
    with pytest.raises(ValueError, match="未知の指標"):
        iqm.evaluate(gray_image, gray_image, metrics=["variance_of_laplacian"])
