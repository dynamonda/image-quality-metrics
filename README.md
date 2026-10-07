# image-quality-metrics

image-quality-metrics は画像の品質評価指標を計測するための python ライブラリです。

## インストール

```bash
uv add git+https://github.com/dynamonda/image-quality-metrics
```

## 使い方

指標には2種類あり、必要な入力が違います。

### 基準画像と比べる（full-reference）

元画像と劣化後の画像を1対1で比較します。

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

### 画像1枚だけで評価する（no-reference）

基準画像が手元にない場合（撮影画像のピント確認など）に使います。

```python
# 鮮鋭度を測る。ピントが合っていれば大きく、ボケていれば小さくなる
iqm.variance_of_laplacian("photo.jpg")  # -> 412.7...

# まとめて計測する
iqm.evaluate_no_reference("photo.jpg")
# -> {"variance_of_laplacian": 412.7...}
```

使える指標名は `iqm.available_metrics()` と `iqm.available_no_reference_metrics()` で取得できます。

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

各指標は「画素値が取り得る最大値」を基準に正規化するため、この値が変わるとスコアも変わります。既定では入力から推定します。

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
| `variance_of_laplacian` | no-reference | ラプラシアン応答の分散で鮮鋭度を測る。大きいほど鮮鋭。 |

**full-reference** は「劣化のない基準画像」が必要な指標、**no-reference** は画像1枚で計算できる指標です。

値は既存の定番実装と機械精度まで一致することを確認しています（`tests/test_reference_values.py`）。PSNR / SSIM は scikit-image、Variance of Laplacian は `scipy.ndimage.laplace` と突き合わせました。

### Variance of Laplacian の注意点

**絶対値には意味がありません。同じ被写体・同じ構図の画像どうしの比較に使ってください。**

ラプラシアンの分散は「画像にエッジがどれだけ含まれるか」を測っているだけなので、被写体の性質に強く左右されます。

- 平坦な被写体（青空、白壁）はピントが合っていても低い値になります
- 細かい模様やノイズの多い画像はボケていても高い値になります
- 画像を縮小すると値が変わります（エッジの密度が変わるため）

「複数枚から最もピントの合ったものを選ぶ」「同じ構図で絞り値を振って比較する」といった用途には有効です。一方で「この画像の鮮鋭度は 300 だから合格」というような絶対的な閾値判定には使えません。

実装上の仕様:

- カーネルは4近傍ラプラシアン（OpenCV の `cv2.Laplacian(ksize=1)` と同じ）
- カラー画像は ITU-R BT.601 で輝度に変換してから計算（アルファは無視）
- スコアは内部で 0-255 スケールに揃えるため、入力が 8bit 整数でも 0-1 の float でも同じ値になります
- 端の扱いは `valid`（他の指標と同様にパディングしない）。OpenCV は既定で端を反射で埋めるため、境界1画素分だけ結果が異なります

## 開発

```bash
uv sync                      # 依存をインストール
uv run pytest                # テスト（docstring の Examples も実行される）
uv run ruff check . --fix    # リント（--fix で自動修正）
uv run ruff format .         # 整形
uv run mypy                  # 型チェック（strict）
uv build                     # dist/ に配布物を生成
```

これらはすべて CI でも実行されます。整形は `ruff format --check` で検査されるため、
コミット前に `uv run ruff format .` を通してください。

### 指標を追加する手順

1. `src/image_quality_metrics/metrics/<指標名>.py` に関数を書く。
   画像の読み込みや検証は `_core.py` の `to_array` / `check_same_shape` /
   `resolve_data_range` / `to_luminance` を使い回す。
2. `evaluate.py` の登録表に1行追加する。種別に対応するものを選ぶ。
   - 基準画像が必要 → `_FULL_REFERENCE_METRICS`
   - 画像1枚で済む → `_NO_REFERENCE_METRICS`
3. `metrics/__init__.py` と `__init__.py` の import / `__all__` に追加する
   （後者に載せたものが公開APIになる）。
4. `tests/test_<指標名>.py` を追加する。
   単調性・スケール不変性・退化した入力での値（一様な画像など）が検算しやすい。
   full-reference なら完全一致時の値と対称性も使える。
5. 既存実装（scikit-image, scipy, OpenCV など）と突き合わせ、一致する値を
   `tests/test_reference_values.py` に固定する。依存は増やさず値だけ埋め込む。

分布系の指標（FID, KID など）は画像セット同士を比較するため、上の2つとは
さらに引数の形が違います。3つ目の登録表を用意してください。

## ライセンス

[MIT License](LICENSE)
