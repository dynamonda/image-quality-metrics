"""複数枚から最もピントの合った画像を選ぶ（no-reference）。

ボケ具合の違う画像を用意し、Variance of Laplacian が最大のものを選ぶ。
この指標は絶対値に意味がないため、同じ被写体どうしの比較に使う。

実行例::

    uv run python examples/pick_sharpest.py            # 合成画像
    uv run python examples/pick_sharpest.py photo.png  # 手元の画像
"""

from __future__ import annotations

import random

from PIL import ImageFilter
from sample_image import load_sample_image

import image_quality_metrics as iqm

BLUR_RADII = (0.0, 0.5, 1.0, 2.0, 4.0)


def main() -> None:
    """ボケ量を変えた候補を作り、鮮鋭度の高い順に並べる。"""
    original = load_sample_image()

    # 連写した写真のように、どれが一番鮮明か分からない状態を再現する。
    radii = list(BLUR_RADII)
    random.Random(0).shuffle(radii)
    candidates = {
        f"shot_{index}": original.filter(ImageFilter.GaussianBlur(radius))
        for index, radius in enumerate(radii)
    }
    true_radius = dict(zip(candidates, radii, strict=True))

    sharpness = {
        name: iqm.variance_of_laplacian(image) for name, image in candidates.items()
    }

    print(f"{'name':<8}  {'blur':>4}  {'sharpness':>10}")
    for name, score in sorted(sharpness.items(), key=lambda item: -item[1]):
        print(f"{name:<8}  {true_radius[name]:>4.1f}  {score:>10.1f}")

    best = max(sharpness, key=sharpness.__getitem__)
    print(f"\n最も鮮明: {best} (blur radius = {true_radius[best]})")


if __name__ == "__main__":
    main()
