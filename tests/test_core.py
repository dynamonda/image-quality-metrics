"""土台 (``_core``) の単体テスト。

ダイナミックレンジの推定ルールは、スコアの値そのものを左右する。
黙って誤った値を返すのが最悪なので、推定境界と「推定できない場合」を
明示的に固定しておく。
"""

from __future__ import annotations

import numpy as np
import pytest

from image_quality_metrics._core import (
    as_channels,
    gaussian_kernel_1d,
    resolve_data_range,
    to_array,
)


def test_data_range_inference() -> None:
    """明示されない場合のダイナミックレンジ推定ルール。"""
    assert resolve_data_range(None, np.array([[0.0, 1.0]])) == 1.0
    assert resolve_data_range(None, np.array([[0.0, 200.0]])) == 255.0
    assert resolve_data_range(None, np.array([[0.0, 1000.0]])) == 65535.0


def test_data_range_uses_maximum_across_images() -> None:
    """2枚を渡した場合は両方の最大値を見る。"""
    assert resolve_data_range(None, np.array([[0.5]]), np.array([[200.0]])) == 255.0


def test_explicit_data_range_wins() -> None:
    assert resolve_data_range(4095, np.array([[0.0, 1.0]])) == 4095.0


def test_negative_pixels_require_explicit_range() -> None:
    """負の値があるとレンジを推定できないので、黙って誤答せず止める。"""
    with pytest.raises(ValueError, match="負の画素値"):
        resolve_data_range(None, np.array([[-1.0, 1.0]]))


def test_invalid_data_range_raises() -> None:
    with pytest.raises(ValueError, match="正の値"):
        resolve_data_range(0, np.array([[0.0, 1.0]]))


def test_to_array_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="2次元"):
        to_array(np.zeros((2, 2, 2, 2)))


def test_to_array_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="空の画像"):
        to_array(np.zeros((0, 4)))


def test_to_array_rejects_zero_channel_image() -> None:
    """チャンネル数0も「空の画像」として弾く。"""
    with pytest.raises(ValueError, match="空の画像"):
        to_array(np.zeros((4, 4, 0)))


def test_to_array_always_returns_float64() -> None:
    """以降の計算を統一するため、入力の dtype に関わらず float64 にする。"""
    assert to_array(np.zeros((4, 4), dtype=np.uint8)).dtype == np.float64


def test_as_channels_adds_channel_axis() -> None:
    assert as_channels(np.zeros((4, 4))).shape == (4, 4, 1)
    assert as_channels(np.zeros((4, 4, 3))).shape == (4, 4, 3)


def test_gaussian_kernel_is_normalized() -> None:
    """重みの総和が1でないと、フィルタが明るさを変えてしまう。"""
    kernel = gaussian_kernel_1d(sigma=1.5, radius=5)

    assert kernel.shape == (11,)
    assert kernel.sum() == pytest.approx(1.0)
    # 中心が最大で、左右対称。
    assert kernel.argmax() == 5
    assert kernel == pytest.approx(kernel[::-1])
