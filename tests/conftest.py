"""テスト全体で共有するフィクスチャ。

``conftest.py`` に置いた fixture は、同じディレクトリ以下のテストから
import なしで引数名だけで使える（pytest の仕組み）。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def rng() -> np.random.Generator:
    """結果を再現できるよう、種を固定した乱数生成器。"""
    return np.random.default_rng(20261008)


@pytest.fixture
def gray_image(rng: np.random.Generator) -> NDArray[np.uint8]:
    """8bit グレースケール画像 (64, 64)。"""
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def color_image(rng: np.random.Generator) -> NDArray[np.uint8]:
    """8bit RGB 画像 (64, 64, 3)。"""
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
