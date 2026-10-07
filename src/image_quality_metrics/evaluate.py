"""複数の指標をまとめて計測する便利関数。

各指標の実装はここには書かない。``metrics/`` にある関数を呼ぶだけに留める
ことで、実装の重複が起きないようにしている。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from image_quality_metrics._core import ImageLike
from image_quality_metrics.metrics.psnr import psnr
from image_quality_metrics.metrics.ssim import ssim

__all__ = ["available_metrics", "evaluate"]


class _FullReferenceMetric(Protocol):
    """基準画像と評価画像を1対1で比べる指標の呼び出し規約。"""

    def __call__(
        self,
        reference: ImageLike,
        test: ImageLike,
        *,
        data_range: float | None = None,
    ) -> float: ...


# 指標の登録表。新しい full-reference 指標を追加するときは、
# ``metrics/`` に関数を書いてここに1行足すだけでよい。
_FULL_REFERENCE_METRICS: dict[str, _FullReferenceMetric] = {
    "psnr": psnr,
    "ssim": ssim,
}


def available_metrics() -> tuple[str, ...]:
    """:func:`evaluate` で指定できる指標名を返す。"""
    return tuple(_FULL_REFERENCE_METRICS)


def evaluate(
    reference: ImageLike,
    test: ImageLike,
    metrics: Iterable[str] | None = None,
    *,
    data_range: float | None = None,
) -> dict[str, float]:
    """複数の指標を一度に計測し、``{指標名: 値}`` を返す。

    Args:
        reference: 基準となる画像。
        test: 評価したい画像。
        metrics: 計測する指標名。``None`` なら利用可能な全指標を計測する。
        data_range: 画素値が取り得る範囲。``None`` なら入力から推定する。

    Returns:
        指標名から値への辞書。``metrics`` を指定した場合はその順序を保つ。

    Raises:
        ValueError: 未知の指標名が指定された場合。

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> image = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        >>> result = evaluate(image, image)
        >>> sorted(result)
        ['psnr', 'ssim']
    """
    requested: Sequence[str] = (
        available_metrics() if metrics is None else tuple(metrics)
    )

    unknown = [name for name in requested if name not in _FULL_REFERENCE_METRICS]
    if unknown:
        raise ValueError(
            f"未知の指標です: {unknown}。"
            f"利用できる指標: {list(available_metrics())}"
        )

    return {
        name: _FULL_REFERENCE_METRICS[name](
            reference, test, data_range=data_range
        )
        for name in requested
    }
