"""すべての指標が共有する土台（画像の読み込み・検証・フィルタ）。

モジュール名を ``_`` で始めているのは「ライブラリ内部の実装であり、
公開APIではない」という意思表示。利用者はここを直接importしない。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

# 利用者が画像として渡せるものの型。パス・PIL画像・numpy配列を受け付ける。
ImageLike: TypeAlias = "str | os.PathLike[str] | Image.Image | NDArray[np.generic]"

# Pillow のモードのうち、そのまま配列化してよいもの。
# これ以外（P, RGBA, CMYK など）は RGB に変換してから扱う。
_PASSTHROUGH_MODES = frozenset({"L", "RGB", "I", "I;16", "F"})


def to_array(image: ImageLike) -> NDArray[np.float64]:
    """任意の入力を ``float64`` の配列 ``(H, W)`` / ``(H, W, C)`` に正規化する。

    値のスケールは変換しない（uint8 なら 0-255 のまま float64 になる）。
    スケールの解釈は :func:`resolve_data_range` が担当する。
    """
    if isinstance(image, np.ndarray):
        array = image
    else:
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.open(os.fspath(image))
        if pil_image.mode not in _PASSTHROUGH_MODES:
            pil_image = pil_image.convert("RGB")
        array = np.asarray(pil_image)

    if array.ndim not in (2, 3):
        raise ValueError(
            f"画像は2次元 (H, W) か3次元 (H, W, C) である必要があります: ndim={array.ndim}"
        )
    if array.ndim == 3 and array.shape[2] == 0:
        raise ValueError("チャンネル数が0の画像は扱えません")
    if array.size == 0:
        raise ValueError("空の画像は扱えません")

    return array.astype(np.float64, copy=False)


def check_same_shape(
    reference: NDArray[np.float64], test: NDArray[np.float64]
) -> None:
    """2枚の画像の形が一致しているか検証する。

    PSNR や SSIM のような full-reference 指標は、画素を1対1で比較するため
    形が違うと計算そのものが定義できない。
    """
    if reference.shape != test.shape:
        raise ValueError(
            "2枚の画像は同じ形でなければなりません: "
            f"reference={reference.shape}, test={test.shape}"
        )


def resolve_data_range(
    data_range: float | None, *images: NDArray[np.float64]
) -> float:
    """画素値が取り得る範囲（ダイナミックレンジ）を決める。

    PSNR/SSIM は「取り得る最大値」を基準に正規化するため、この値が
    変わるとスコアも変わる。``None`` のときは入力から推定する。
    """
    if data_range is not None:
        if data_range <= 0:
            raise ValueError(f"data_range は正の値でなければなりません: {data_range}")
        return float(data_range)

    maximum = max(float(image.max()) for image in images)
    minimum = min(float(image.min()) for image in images)
    if minimum < 0.0:
        raise ValueError(
            "負の画素値を含む画像では data_range を自動推定できません。"
            "明示的に data_range を指定してください。"
        )
    # 0-1 に収まっていれば float 画像、それ以外は 8bit/16bit 整数画像とみなす。
    if maximum <= 1.0:
        return 1.0
    if maximum <= 255.0:
        return 255.0
    return 65535.0


def as_channels(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(H, W)`` を ``(H, W, 1)`` に揃え、チャンネル単位の処理を書きやすくする。"""
    if array.ndim == 2:
        return array[:, :, np.newaxis]
    return array


def gaussian_kernel_1d(sigma: float, radius: int) -> NDArray[np.float64]:
    """正規化された1次元ガウシアンカーネルを返す。"""
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def filter_2d_valid(
    plane: NDArray[np.float64], kernel_1d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """2次元画像に分離可能フィルタを ``valid`` モードで畳み込む。

    2次元ガウシアンは「横方向の1次元ガウシアン」と「縦方向の1次元ガウシアン」に
    分解できる（分離可能性）。そのため O(K^2) ではなく O(K) で済む。

    ``valid`` モード（端を落とす）にしているのは、端をパディングで埋めると
    存在しない画素を根拠にスコアが動いてしまうため。
    """
    windows = np.lib.stride_tricks.sliding_window_view(plane, kernel_1d.size, axis=-1)
    horizontal = windows @ kernel_1d
    windows = np.lib.stride_tricks.sliding_window_view(
        horizontal, kernel_1d.size, axis=-2
    )
    return windows @ kernel_1d
