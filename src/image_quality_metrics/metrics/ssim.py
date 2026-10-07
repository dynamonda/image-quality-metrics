"""SSIM (Structural Similarity Index Measure)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from image_quality_metrics._core import (
    ImageLike,
    as_channels,
    check_same_shape,
    filter_2d_valid,
    gaussian_kernel_1d,
    resolve_data_range,
    to_array,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ssim", "ssim_map"]

# Wang et al. (2004) の原論文で使われている既定値。
_DEFAULT_WINDOW_SIZE = 11
_DEFAULT_SIGMA = 1.5
_K1 = 0.01
_K2 = 0.03


def ssim(
    reference: ImageLike,
    test: ImageLike,
    *,
    data_range: float | None = None,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    sigma: float = _DEFAULT_SIGMA,
) -> float:
    """2枚の画像間の SSIM を返す。

    PSNR が画素ごとの誤差しか見ないのに対し、SSIM は局所的な輝度・コント
    ラスト・構造の3要素を比較するため、人間の知覚により近いとされる。
    値域は ``[-1, 1]`` で、1 に近いほど似ている。

    カラー画像の場合はチャンネルごとに計算し、その平均を返す。

    Args:
        reference: 基準となる画像。
        test: 評価したい画像。``reference`` と同じ形である必要がある。
        data_range: 画素値が取り得る範囲。``None`` なら入力から推定する。
        window_size: 局所統計を取るガウシアン窓の一辺（奇数）。
        sigma: ガウシアン窓の標準偏差。

    Returns:
        SSIM マップ全体の平均値（mean SSIM）。

    Raises:
        ValueError: 画像の形が異なる、窓サイズが不正、または画像が窓より
            小さい場合。

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> image = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        >>> round(ssim(image, image), 6)
        1.0
    """
    return float(
        ssim_map(
            reference,
            test,
            data_range=data_range,
            window_size=window_size,
            sigma=sigma,
        ).mean()
    )


def ssim_map(
    reference: ImageLike,
    test: ImageLike,
    *,
    data_range: float | None = None,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    sigma: float = _DEFAULT_SIGMA,
) -> NDArray[np.float64]:
    """SSIM を平均せず、画素ごとの類似度マップとして返す。

    「どこが劣化しているか」を可視化したいときに使う。形は入力より窓の分だけ
    小さい ``(H - window_size + 1, W - window_size + 1, C)``。

    引数の意味は :func:`ssim` と同じ。
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size は正の奇数でなければなりません: {window_size}")
    if sigma <= 0:
        raise ValueError(f"sigma は正の値でなければなりません: {sigma}")

    reference_array = to_array(reference)
    test_array = to_array(test)
    check_same_shape(reference_array, test_array)
    resolved_range = resolve_data_range(data_range, reference_array, test_array)

    height, width = reference_array.shape[:2]
    if min(height, width) < window_size:
        raise ValueError(
            f"画像が窓より小さいため SSIM を計算できません: "
            f"画像サイズ=({height}, {width}), window_size={window_size}"
        )

    reference_planes = as_channels(reference_array)
    test_planes = as_channels(test_array)
    kernel = gaussian_kernel_1d(sigma, window_size // 2)

    # 安定化定数。分母が 0 に近いときの数値的な破綻を防ぐ役割で、
    # ダイナミックレンジに対する相対値として定義される。
    c1 = (_K1 * resolved_range) ** 2
    c2 = (_K2 * resolved_range) ** 2

    channel_maps = [
        _ssim_single_plane(
            reference_planes[:, :, index], test_planes[:, :, index], kernel, c1, c2
        )
        for index in range(reference_planes.shape[2])
    ]
    return np.stack(channel_maps, axis=-1)


def _ssim_single_plane(
    reference: NDArray[np.float64],
    test: NDArray[np.float64],
    kernel: NDArray[np.float64],
    c1: float,
    c2: float,
) -> NDArray[np.float64]:
    """単一チャンネルに対する SSIM マップ。"""
    # 局所平均。ガウシアン窓で重み付けした「その近傍の明るさ」。
    mean_reference = filter_2d_valid(reference, kernel)
    mean_test = filter_2d_valid(test, kernel)

    # 局所分散・共分散は E[X^2] - E[X]^2 の形で求める。
    variance_reference = (
        filter_2d_valid(reference * reference, kernel) - mean_reference**2
    )
    variance_test = filter_2d_valid(test * test, kernel) - mean_test**2
    covariance = filter_2d_valid(reference * test, kernel) - mean_reference * mean_test

    numerator = (2.0 * mean_reference * mean_test + c1) * (2.0 * covariance + c2)
    denominator = (mean_reference**2 + mean_test**2 + c1) * (
        variance_reference + variance_test + c2
    )
    return numerator / denominator
