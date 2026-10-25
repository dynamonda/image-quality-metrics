"""iqm コマンド（cli.main）のテスト。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

import image_quality_metrics as iqm
from image_quality_metrics.cli import main

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _save(array: NDArray[np.uint8], path: Path) -> str:
    Image.fromarray(array).save(path)
    return str(path)


@pytest.fixture
def image_pair(
    color_image: NDArray[np.uint8], rng: np.random.Generator, tmp_path: Path
) -> tuple[str, str]:
    """基準画像と、ノイズを加えた評価画像のパス。"""
    noisy = np.clip(color_image + rng.normal(0, 10, color_image.shape), 0, 255)
    return (
        _save(color_image, tmp_path / "reference.png"),
        _save(noisy.astype(np.uint8), tmp_path / "test.png"),
    )


def _parse_output(text: str) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in (line.split() for line in text.splitlines())
    }


def test_two_images_run_full_reference(
    image_pair: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(list(image_pair)) == 0

    scores = _parse_output(capsys.readouterr().out)
    assert list(scores) == list(iqm.available_metrics())
    assert scores["psnr"] == pytest.approx(iqm.psnr(*image_pair), abs=1e-4)


def test_one_image_runs_no_reference(
    image_pair: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([image_pair[0]]) == 0

    scores = _parse_output(capsys.readouterr().out)
    assert list(scores) == list(iqm.available_no_reference_metrics())


def test_metrics_option_selects_and_orders(
    image_pair: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*image_pair, "--metrics", "ssim, psnr"]) == 0

    assert list(_parse_output(capsys.readouterr().out)) == ["ssim", "psnr"]


def test_data_range_is_passed_through(
    image_pair: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*image_pair, "-m", "psnr", "--data-range", "65535"]) == 0

    psnr = _parse_output(capsys.readouterr().out)["psnr"]
    assert psnr == pytest.approx(iqm.psnr(*image_pair, data_range=65535), abs=1e-4)


def test_unknown_metric_reports_error(
    image_pair: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*image_pair, "-m", "lpips"]) == 1

    assert "未知の指標" in capsys.readouterr().err


def test_missing_file_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "missing.png")]) == 1

    assert capsys.readouterr().err.startswith("iqm: error:")


def test_three_images_are_rejected(image_pair: tuple[str, str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([*image_pair, image_pair[0]])

    assert excinfo.value.code == 2
