"""PSNR (Peak Signal-to-Noise Ratio)."""

from __future__ import annotations

import numpy as np

from image_quality_metrics._core import (
    ImageLike,
    check_same_shape,
    resolve_data_range,
    to_array,
)

__all__ = ["psnr"]


def psnr(
    reference: ImageLike,
    test: ImageLike,
    *,
    data_range: float | None = None,
) -> float:
    """2枚の画像間の PSNR を dB で返す。

    画素ごとの二乗誤差（MSE）をダイナミックレンジで正規化し、対数スケールに
    したもの。値が大きいほど元画像に近い。目安として 30dB 以上なら劣化は
    ほとんど目立たない。

    計算は全チャンネルをまとめて行う（カラー画像なら RGB 全体の MSE）。

    Args:
        reference: 基準となる画像。パス・PIL画像・numpy配列のいずれか。
        test: 評価したい画像。``reference`` と同じ形である必要がある。
        data_range: 画素値が取り得る範囲。``None`` なら入力から推定する
            (0-1 の float 画像なら ``1.0``、8bit 画像なら ``255.0``)。

    Returns:
        PSNR の値（dB）。2枚が完全に一致する場合は ``float("inf")``。

    Raises:
        ValueError: 2枚の画像の形が異なる場合。

    Examples:
        >>> import numpy as np
        >>> image = np.zeros((4, 4), dtype=np.uint8)
        >>> psnr(image, image)
        inf
    """
    reference_array = to_array(reference)
    test_array = to_array(test)
    check_same_shape(reference_array, test_array)
    resolved_range = resolve_data_range(data_range, reference_array, test_array)

    mean_squared_error = float(np.mean((reference_array - test_array) ** 2))
    # 完全一致だと MSE が 0 になり、対数が発散する。これは「劣化なし」を
    # 意味するので inf を返すのが慣例。
    if mean_squared_error == 0.0:
        return float("inf")

    return float(10.0 * np.log10(resolved_range**2 / mean_squared_error))
