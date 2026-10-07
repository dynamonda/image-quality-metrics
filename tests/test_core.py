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
    correlate_2d_valid,
    gaussian_kernel_1d,
    resolve_data_range,
    to_array,
    to_luminance,
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


def test_to_luminance_uses_bt601_weights() -> None:
    """BT.601 の係数 (0.299, 0.587, 0.114) で重み付けされる。"""
    red = np.zeros((1, 1, 3))
    red[0, 0, 0] = 255.0
    green = np.zeros((1, 1, 3))
    green[0, 0, 1] = 255.0
    blue = np.zeros((1, 1, 3))
    blue[0, 0, 2] = 255.0

    assert to_luminance(red)[0, 0] == pytest.approx(255.0 * 0.299)
    assert to_luminance(green)[0, 0] == pytest.approx(255.0 * 0.587)
    assert to_luminance(blue)[0, 0] == pytest.approx(255.0 * 0.114)


def test_to_luminance_weights_sum_to_one() -> None:
    """グレー（R=G=B）がそのままの明るさで残ることの確認。"""
    gray = np.full((2, 2, 3), 200.0)

    assert to_luminance(gray) == pytest.approx(np.full((2, 2), 200.0))


def test_to_luminance_shapes() -> None:
    """2次元はそのまま、3次元は (H, W) に落ちる。"""
    assert to_luminance(np.zeros((4, 4))).shape == (4, 4)
    assert to_luminance(np.zeros((4, 4, 1))).shape == (4, 4)
    assert to_luminance(np.zeros((4, 4, 3))).shape == (4, 4)
    assert to_luminance(np.zeros((4, 4, 4))).shape == (4, 4)


def test_to_luminance_rejects_unsupported_channels() -> None:
    with pytest.raises(ValueError, match="輝度に変換できるのは"):
        to_luminance(np.zeros((4, 4, 2)))


def test_correlate_2d_valid_shape_and_value() -> None:
    """端を落とす valid モードなので、カーネル分だけ小さくなる。"""
    plane = np.arange(25, dtype=np.float64).reshape(5, 5)
    kernel = np.ones((3, 3))

    result = correlate_2d_valid(plane, kernel)

    assert result.shape == (3, 3)
    # 3x3 の総和。中心が 6 の窓なら 0+1+2+5+6+7+10+11+12 = 54。
    assert result[0, 0] == pytest.approx(54.0)


def test_correlate_2d_valid_does_not_flip_kernel() -> None:
    """相関なのでカーネルは反転しない（畳み込みとの違い）。"""
    plane = np.zeros((3, 3))
    plane[0, 0] = 1.0
    kernel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # 反転しないなら、立っている左上の画素にカーネルの左上 (1.0) が当たる。
    assert correlate_2d_valid(plane, kernel)[0, 0] == pytest.approx(1.0)


def test_gaussian_kernel_is_normalized() -> None:
    """重みの総和が1でないと、フィルタが明るさを変えてしまう。"""
    kernel = gaussian_kernel_1d(sigma=1.5, radius=5)

    assert kernel.shape == (11,)
    assert kernel.sum() == pytest.approx(1.0)
    # 中心が最大で、左右対称。
    assert kernel.argmax() == 5
    assert kernel == pytest.approx(kernel[::-1])
