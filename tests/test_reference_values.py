"""既存の定番実装と答えが一致することを固定するリグレッションテスト。

ここに並ぶ期待値は既存実装で算出し、本実装と機械精度まで一致することを
確認したうえで埋め込んだもの。

値を直接埋め込んでいるのは、テストのために scikit-image や scipy を依存に
加えたくないため。実装をリファクタしたときに計算式が静かに壊れていないか、
これらの数点だけで検出できる。

PSNR / SSIM は scikit-image
(``peak_signal_noise_ratio`` / ``structural_similarity``) で算出した。
SSIM の呼び出し条件（原論文と同じ設定）:
    gaussian_weights=True, sigma=1.5, use_sample_covariance=False

Variance of Laplacian は scipy (``scipy.ndimage.laplace``) で算出した。
本実装は valid モードなので、scipy 側は ``mode="reflect"`` で計算してから
``[1:-1, 1:-1]`` に切り出して突き合わせている。
"""

from __future__ import annotations

import numpy as np
import pytest

from image_quality_metrics import psnr, ssim, variance_of_laplacian

_SKIMAGE_GRAY_PSNR = 24.832391675824894
_SKIMAGE_GRAY_SSIM = 0.9798976206239147
_SKIMAGE_COLOR_PSNR = 24.934517677250344
_SKIMAGE_COLOR_SSIM = 0.9800874523257604
_SCIPY_GRAY_VOL = 110131.63365072368
_SCIPY_COLOR_VOL = 46614.75766626878


def _noisy_pair(seed: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """期待値を算出したときと同一の入力を再現する。

    乱数の呼び出し順まで一致させる必要があるので、この関数の中身は変えないこと。
    """
    rng = np.random.default_rng(seed)
    reference = rng.integers(0, 256, shape).astype(np.float64)
    test = np.clip(reference + rng.normal(0, 15, shape), 0, 255)
    return reference, test


def test_gray_psnr_matches_skimage() -> None:
    reference, test = _noisy_pair(12345, (64, 64))

    assert psnr(reference, test, data_range=255) == pytest.approx(
        _SKIMAGE_GRAY_PSNR, rel=1e-12
    )


def test_gray_ssim_matches_skimage() -> None:
    reference, test = _noisy_pair(12345, (64, 64))

    assert ssim(reference, test, data_range=255) == pytest.approx(
        _SKIMAGE_GRAY_SSIM, rel=1e-12
    )


def test_color_psnr_matches_skimage() -> None:
    reference, test = _noisy_pair(54321, (64, 64, 3))

    assert psnr(reference, test, data_range=255) == pytest.approx(
        _SKIMAGE_COLOR_PSNR, rel=1e-12
    )


def test_color_ssim_matches_skimage() -> None:
    reference, test = _noisy_pair(54321, (64, 64, 3))

    assert ssim(reference, test, data_range=255) == pytest.approx(
        _SKIMAGE_COLOR_SSIM, rel=1e-12
    )


def _single_image(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """期待値を算出したときと同一の入力を再現する（無参照指標用）。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape).astype(np.float64)


def test_gray_variance_of_laplacian_matches_scipy() -> None:
    image = _single_image(777, (64, 64))

    assert variance_of_laplacian(image, data_range=255) == pytest.approx(
        _SCIPY_GRAY_VOL, rel=1e-12
    )


def test_color_variance_of_laplacian_matches_scipy() -> None:
    """カラーは輝度へ落としてから計算される。"""
    image = _single_image(888, (64, 64, 3))

    assert variance_of_laplacian(image, data_range=255) == pytest.approx(
        _SCIPY_COLOR_VOL, rel=1e-12
    )
