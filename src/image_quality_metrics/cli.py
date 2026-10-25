"""コマンドラインから指標を計測する入口（``iqm`` コマンド）。

渡された画像の枚数で計測の種類を切り替える。2枚なら1枚目を基準画像とした
full-reference、1枚なら no-reference。指標の計算そのものは
:mod:`image_quality_metrics.evaluate` に任せ、ここでは引数の解釈と出力だけを行う。
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from image_quality_metrics.evaluate import (
    available_metrics,
    available_no_reference_metrics,
    evaluate,
    evaluate_no_reference,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]

_FULL_REFERENCE_IMAGE_COUNT = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iqm",
        description=(
            "画像の品質評価指標を計測する。"
            "2枚渡すと1枚目を基準に比較し、1枚だけなら単体で評価する。"
        ),
        epilog=(
            f"2枚のときの指標: {', '.join(available_metrics())} / "
            f"1枚のときの指標: {', '.join(available_no_reference_metrics())}"
        ),
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="評価する画像。比較する場合は 基準画像 評価画像 の順に渡す。",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        help="計測する指標名をカンマ区切りで指定する（省略時は全指標）。",
    )
    parser.add_argument(
        "--data-range",
        type=float,
        help="画素値が取り得る範囲（例: 12bit なら 4095）。省略時は入力から推定する。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """コマンドを実行し、終了コードを返す。

    Args:
        argv: コマンドライン引数（プログラム名を除く）。``None`` なら
            ``sys.argv`` を使う。

    Returns:
        成功なら 0、計測できなかった場合は 1。引数の誤りは argparse が
        終了コード 2 で終了させる。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if len(args.images) > _FULL_REFERENCE_IMAGE_COUNT:
        parser.error(f"画像は1枚か2枚で指定してください: {len(args.images)} 枚")

    metrics = (
        None
        if args.metrics is None
        else [name.strip() for name in args.metrics.split(",") if name.strip()]
    )

    try:
        if len(args.images) == _FULL_REFERENCE_IMAGE_COUNT:
            reference, test = args.images
            scores = evaluate(reference, test, metrics, data_range=args.data_range)
        else:
            (image,) = args.images
            scores = evaluate_no_reference(image, metrics, data_range=args.data_range)
    # 画像が開けない（存在しない・形式不明）場合は OSError、形の不一致や
    # 未知の指標名は ValueError になる。どちらもトレースバックではなく
    # 1行のメッセージで伝える。
    except (OSError, ValueError) as error:
        print(f"iqm: error: {error}", file=sys.stderr)
        return 1

    width = max(len(name) for name in scores)
    for name, value in scores.items():
        print(f"{name:<{width}}  {value:.4f}")
    return 0
