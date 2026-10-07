"""Variance of Laplacian（ラプラシアンの分散）。"""

from __future__ import annotations

import numpy as np

from image_quality_metrics._core import (
    ImageLike,
    correlate_2d_valid,
    resolve_data_range,
    to_array,
    to_luminance,
)

__all__ = ["variance_of_laplacian"]

# 4近傍ラプラシアン。OpenCV の cv2.Laplacian(ksize=1) と同じカーネル。
_LAPLACIAN_KERNEL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

_KERNEL_SIZE = 3

# スコアを報告するスケール。入力が 0-1 の float でも 8bit 整数でも同じ値に
# なるよう、内部で 0-255 に揃えてから分散を取る。広く引用されている
# 「閾値100前後でボケ判定」という経験則が 8bit 画像前提のため、それに合わせる。
_SCORE_SCALE = 255.0


def variance_of_laplacian(
    image: ImageLike,
    *,
    data_range: float | None = None,
) -> float:
    """画像の鮮鋭度を Variance of Laplacian で返す。

    ラプラシアンは2次微分であり、エッジで大きく応答する。その応答の分散を
    取ると「画像にエッジがどれだけ含まれるか」の指標になる。ピントが合って
    いれば大きく、ボケていれば小さくなる。

    基準画像を必要としない no-reference 指標なので、画像1枚で計算できる。

    値は 0-255 スケールに揃えてから算出するため、入力が 8bit 整数でも
    0-1 の float でも同じスコアになる。カラー画像は先に輝度へ変換する。

    Note:
        絶対値には意味がなく、**同じ被写体・同じ構図の画像どうしの比較**に
        使うこと。平坦な被写体（青空、白壁）はピントが合っていても低い値に
        なり、細かい模様やノイズが多い画像はボケていても高い値になる。

    Args:
        image: 評価したい画像。パス・PIL画像・numpy配列のいずれか。
        data_range: 画素値が取り得る範囲。``None`` なら入力から推定する。

    Returns:
        ラプラシアン応答の分散。大きいほど鮮鋭。

    Raises:
        ValueError: 画像が 3x3 より小さい場合。

    Examples:
        >>> import numpy as np
        >>> flat = np.full((16, 16), 128, dtype=np.uint8)
        >>> variance_of_laplacian(flat)
        0.0
    """
    array = to_array(image)
    resolved_range = resolve_data_range(data_range, array)

    height, width = array.shape[:2]
    if min(height, width) < _KERNEL_SIZE:
        raise ValueError(
            "Variance of Laplacian には 3x3 以上の画像が必要です: "
            f"画像サイズ=({height}, {width})"
        )

    luminance = to_luminance(array) * (_SCORE_SCALE / resolved_range)
    response = correlate_2d_valid(luminance, _LAPLACIAN_KERNEL)
    return float(response.var())
