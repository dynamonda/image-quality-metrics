# image-quality-metrics

image-quality-metrics は画像の品質評価指標を計測するための python ライブラリです。

## インストール

```bash
uv add git+https://github.com/dynamonda/image-quality-metrics
```

## 使い方

```python
import image_quality_metrics as iqm

# 個別に計測する
iqm.psnr("original.png", "compressed.png")  # -> 31.24...
iqm.ssim("original.png", "compressed.png")  # -> 0.9412...

# まとめて計測する
iqm.evaluate("original.png", "compressed.png")
# -> {"psnr": 31.24..., "ssim": 0.9412...}

# 指標を選ぶ（指定した順序で返る）
iqm.evaluate("original.png", "compressed.png", metrics=["ssim"])
# -> {"ssim": 0.9412...}
```

画像はファイルパス・`PIL.Image`・`numpy.ndarray` のいずれでも渡せます。混ぜても構いません。

```python
import numpy as np
from PIL import Image

iqm.psnr(np.asarray(Image.open("a.png")), "b.png")
```

劣化箇所を可視化したい場合は、平均を取る前の SSIM マップが取れます。

```python
heatmap = iqm.ssim_map("original.png", "compressed.png")  # (H-10, W-10, C)
```

### ダイナミックレンジ

PSNR / SSIM は「画素値が取り得る最大値」を基準に正規化するため、この値が変わるとスコアも変わります。既定では入力から推定します。

| 入力の最大値 | 推定される `data_range` |
| --- | --- |
| `<= 1.0` | `1.0`（0-1 正規化された float 画像） |
| `<= 255.0` | `255.0`（8bit 画像） |
| それ以上 | `65535.0`（16bit 画像） |

意図した推定にならない場合（暗い画像、12bit RAW など）は明示してください。

```python
iqm.psnr(reference, test, data_range=4095)  # 12bit
```

負の画素値を含む場合は推定できないため、`data_range` の指定が必須です。

## 対応している指標

| 指標 | 種別 | 説明 |
| --- | --- | --- |
| `psnr` | full-reference | 画素ごとの二乗誤差ベース。大きいほど良い（単位 dB）。 |
| `ssim` | full-reference | 局所的な輝度・コントラスト・構造を比較。`[-1, 1]`、1 に近いほど良い。 |

full-reference は「劣化のない基準画像」が必要な指標です。値は scikit-image の実装と機械精度まで一致することを確認しています（`tests/test_reference_values.py`）。

## 開発

```bash
uv sync              # 依存をインストール
uv run pytest        # テスト（docstring の Examples も実行される）
uv build             # dist/ に配布物を生成
```

### 指標を追加する手順

1. `src/image_quality_metrics/metrics/<指標名>.py` に関数を書く。
   画像の読み込みや検証は `_core.py` の `to_array` / `check_same_shape` /
   `resolve_data_range` を使い回す。
2. full-reference 指標なら `evaluate.py` の `_FULL_REFERENCE_METRICS` に1行追加する。
3. `__init__.py` の import と `__all__` に追加する（ここに載せたものが公開APIになる）。
4. `tests/test_<指標名>.py` を追加する。
   完全一致時の値・単調性・対称性・スケール不変性あたりが検算しやすい。

no-reference 指標（NIQE, BRISQUE など）や分布系の指標（FID など）は引数の形が
違うため、`_FULL_REFERENCE_METRICS` とは別の登録表を用意してください。

## ライセンス

未定。
