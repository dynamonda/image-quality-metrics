# 使用例

リポジトリのルートで実行します。引数を省略すると合成画像を使うため、画像を用意しなくても動きます。

```bash
uv run python examples/compare_jpeg_quality.py            # 合成画像
uv run python examples/compare_jpeg_quality.py photo.png  # 手元の画像
```

| ファイル | 内容 | 使っている API |
| --- | --- | --- |
| [`compare_jpeg_quality.py`](compare_jpeg_quality.py) | JPEG の品質設定ごとに PSNR / SSIM を表にする | `evaluate` |
| [`pick_sharpest.py`](pick_sharpest.py) | ボケ具合の違う画像から最も鮮明なものを選ぶ | `variance_of_laplacian` |
| [`ssim_heatmap.py`](ssim_heatmap.py) | 一部だけ劣化させた画像の SSIM マップを PNG で保存する | `ssim_map` |

`sample_image.py` は入力画像を用意する共通処理です（例そのものではありません）。

`ssim_heatmap.py` の出力は `examples/output/` に保存されます（git 管理外）。白いほど元画像に近く、黒いほど劣化しています。
