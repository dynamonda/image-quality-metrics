"""複数の指標をまとめて計測する便利関数。

各指標の実装はここには書かない。``metrics/`` にある関数を呼ぶだけに留める
ことで、実装の重複が起きないようにしている。

登録表を指標の種別ごとに分けているのは、種別によって必要な入力が違うため。
full-reference は劣化のない基準画像を要求し、no-reference は画像1枚で済む。
同じ表に混ぜると、呼び出し側が「この指標には基準画像が要るのか」を
実行してみるまで判断できなくなる。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

from image_quality_metrics._core import ImageLike
from image_quality_metrics.metrics.psnr import psnr
from image_quality_metrics.metrics.ssim import ssim
from image_quality_metrics.metrics.variance_of_laplacian import variance_of_laplacian

__all__ = [
    "available_metrics",
    "available_no_reference_metrics",
    "evaluate",
    "evaluate_no_reference",
]


class _FullReferenceMetric(Protocol):
    """基準画像と評価画像を1対1で比べる指標の呼び出し規約。"""

    def __call__(
        self,
        reference: ImageLike,
        test: ImageLike,
        *,
        data_range: float | None = None,
    ) -> float: ...


class _NoReferenceMetric(Protocol):
    """画像1枚だけで品質を推定する指標の呼び出し規約。"""

    def __call__(
        self,
        image: ImageLike,
        *,
        data_range: float | None = None,
    ) -> float: ...


# full-reference 指標の登録表。追加するときは ``metrics/`` に関数を書いて
# ここに1行足すだけでよい。
_FULL_REFERENCE_METRICS: dict[str, _FullReferenceMetric] = {
    "psnr": psnr,
    "ssim": ssim,
}

# no-reference 指標の登録表。
_NO_REFERENCE_METRICS: dict[str, _NoReferenceMetric] = {
    "variance_of_laplacian": variance_of_laplacian,
}


def available_metrics() -> tuple[str, ...]:
    """:func:`evaluate` で指定できる full-reference 指標名を返す。"""
    return tuple(_FULL_REFERENCE_METRICS)


def available_no_reference_metrics() -> tuple[str, ...]:
    """:func:`evaluate_no_reference` で指定できる指標名を返す。"""
    return tuple(_NO_REFERENCE_METRICS)


def _select(
    metrics: Iterable[str] | None, table: Mapping[str, object]
) -> tuple[str, ...]:
    """計測する指標名を確定し、未知の名前があれば弾く。

    ``None`` のときは登録表の全件。指定された場合はその順序を保つ。
    """
    requested = tuple(table) if metrics is None else tuple(metrics)

    unknown = [name for name in requested if name not in table]
    if unknown:
        raise ValueError(f"未知の指標です: {unknown}。利用できる指標: {list(table)}")
    return requested


def evaluate(
    reference: ImageLike,
    test: ImageLike,
    metrics: Iterable[str] | None = None,
    *,
    data_range: float | None = None,
) -> dict[str, float]:
    """基準画像と比較する指標を一度に計測し、``{指標名: 値}`` を返す。

    Args:
        reference: 劣化のない基準画像。
        test: 評価したい画像。``reference`` と同じ形である必要がある。
        metrics: 計測する指標名。``None`` なら
            :func:`available_metrics` の全指標を計測する。
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
    return {
        name: _FULL_REFERENCE_METRICS[name](reference, test, data_range=data_range)
        for name in _select(metrics, _FULL_REFERENCE_METRICS)
    }


def evaluate_no_reference(
    image: ImageLike,
    metrics: Iterable[str] | None = None,
    *,
    data_range: float | None = None,
) -> dict[str, float]:
    """基準画像を必要としない指標を一度に計測し、``{指標名: 値}`` を返す。

    Args:
        image: 評価したい画像。
        metrics: 計測する指標名。``None`` なら
            :func:`available_no_reference_metrics` の全指標を計測する。
        data_range: 画素値が取り得る範囲。``None`` なら入力から推定する。

    Returns:
        指標名から値への辞書。``metrics`` を指定した場合はその順序を保つ。

    Raises:
        ValueError: 未知の指標名が指定された場合。

    Examples:
        >>> import numpy as np
        >>> flat = np.full((16, 16), 128, dtype=np.uint8)
        >>> evaluate_no_reference(flat, data_range=255)
        {'variance_of_laplacian': 0.0}
    """
    return {
        name: _NO_REFERENCE_METRICS[name](image, data_range=data_range)
        for name in _select(metrics, _NO_REFERENCE_METRICS)
    }
