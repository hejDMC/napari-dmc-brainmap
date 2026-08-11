from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from napari_dmc_brainmap.stitching import tiff_output


@pytest.fixture
def scratch_root(monkeypatch, tmp_path):
    root = tmp_path / "app-cache" / "stitching" / "scratch"
    monkeypatch.setattr(tiff_output, "stitching_scratch_root", lambda: root)
    return root


def test_stitching_scratch_is_sibling_of_atlas_cache(monkeypatch, tmp_path):
    app_cache = tmp_path / "app-cache"
    monkeypatch.setattr(tiff_output, "app_cache_root", lambda: app_cache)

    result = tiff_output.stitching_scratch_root()

    assert result == app_cache / "stitching" / "scratch"
    assert result.is_dir()


def test_insufficient_cache_space_fails_before_creating_output(
    monkeypatch,
    scratch_root,
    tmp_path,
):
    scratch_root.mkdir(parents=True)
    monkeypatch.setattr(
        tiff_output,
        "disk_usage",
        lambda _: SimpleNamespace(free=1024),
    )
    destination = tmp_path / "section.tif"

    with pytest.raises(
        OSError,
        match="Not enough local application-cache space",
    ):
        with tiff_output.compressed_stitch_canvas(destination, (2048, 2048)):
            pass

    assert not destination.exists()
    assert not list(scratch_root.iterdir())


def test_compressed_canvas_is_lossless_and_published_only_when_complete(
    scratch_root,
    tmp_path,
):
    scratch_root.mkdir(parents=True)
    destination = tmp_path / "section.tif"
    old_image = np.full((4, 5), 9, dtype=np.uint16)
    expected = np.arange(20, dtype=np.uint16).reshape(4, 5)
    tifffile.imwrite(destination, old_image, photometric="minisblack")

    with tiff_output.compressed_stitch_canvas(
        destination,
        expected.shape,
    ) as canvas:
        canvas[:] = expected
        np.testing.assert_array_equal(tifffile.imread(destination), old_image)

    np.testing.assert_array_equal(tifffile.imread(destination), expected)
    with tifffile.TiffFile(destination) as tif:
        assert tif.pages[0].tags["Compression"].value in (8, 32946)
        assert tif.pages[0].tags["Predictor"].value == 2
    assert not list(scratch_root.iterdir())
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))


def test_stitching_failure_preserves_existing_output_and_removes_scratch(
    scratch_root,
    tmp_path,
):
    scratch_root.mkdir(parents=True)
    destination = tmp_path / "section.tif"
    expected = np.full((2, 2), 11, dtype=np.uint16)
    tifffile.imwrite(destination, expected, photometric="minisblack")

    with pytest.raises(RuntimeError, match="simulated stitching failure"):
        with tiff_output.compressed_stitch_canvas(
            destination,
            expected.shape,
        ) as canvas:
            canvas[:] = 99
            raise RuntimeError("simulated stitching failure")

    np.testing.assert_array_equal(tifffile.imread(destination), expected)
    assert not list(scratch_root.iterdir())
    assert not list(tmp_path.glob(".*.partial"))


def test_canvas_always_uses_application_cache(
    scratch_root,
    tmp_path,
):
    scratch_root.mkdir(parents=True)
    destination = tmp_path / "network-output.tif"
    scratch_paths: list[Path] = []

    with tiff_output.compressed_stitch_canvas(destination, (2, 2)) as canvas:
        canvas[:] = 7
        scratch_paths.append(Path(canvas.filename))
        assert scratch_paths[0].parent.parent == scratch_root
        assert scratch_paths[0].parent != destination.parent

    np.testing.assert_array_equal(
        tifffile.imread(destination),
        np.full((2, 2), 7, dtype=np.uint16),
    )
    assert not scratch_paths[0].exists()


def test_atomic_write_preserves_existing_file_and_removes_partial_on_failure(
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "section.tif"
    expected = np.full((2, 2), 3, dtype=np.uint16)
    tifffile.imwrite(destination, expected, photometric="minisblack")

    def fail_write(*args, **kwargs):
        raise OSError("simulated interrupted transfer")

    monkeypatch.setattr(tiff_output.tifffile, "imwrite", fail_write)

    with pytest.raises(OSError, match="interrupted transfer"):
        tiff_output.write_tiff_atomically(
            destination,
            np.zeros((2, 2), dtype=np.uint16),
        )

    np.testing.assert_array_equal(tifffile.imread(destination), expected)
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))
