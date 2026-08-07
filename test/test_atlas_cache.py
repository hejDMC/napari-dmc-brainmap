from pathlib import Path
from types import SimpleNamespace

import numpy as np

from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.AtlasModel import (
    AtlasModel,
)
from napari_dmc_brainmap.results.probe_vis.probe_vis.view.ProbeVisualizer import (
    ProbeVisualizer,
)
from napari_dmc_brainmap.segment.processing import atlas_utils
from napari_dmc_brainmap.utils import atlas_cache


def _fake_atlas(
    atlas_root: Path,
    *,
    version: tuple[int, int] = (1, 2),
    resolution: tuple[int, int, int] = (25, 25, 25),
) -> SimpleNamespace:
    template = np.asarray(
        [
            [[100, 200], [300, 400]],
            [[500, 600], [700, 800]],
        ],
        dtype=np.uint16,
    )
    annotation = np.asarray(
        [
            [[0, 1], [2, 0]],
            [[3, 4], [0, 0]],
        ],
        dtype=np.uint32,
    )
    return SimpleNamespace(
        atlas_name="example_mouse_25um",
        local_version=version,
        metadata={"name": "example_mouse"},
        resolution=resolution,
        shape=template.shape,
        root_dir=atlas_root,
        template=template,
        annotation=annotation,
    )


def test_cache_path_is_keyed_by_atlas_version_and_resolution(
    tmp_path: Path,
) -> None:
    atlas = _fake_atlas(tmp_path / "brainglobe" / "example_v1.2")
    cache_root = tmp_path / "app-cache"

    first = atlas_cache.atlas_cache_dir(atlas, cache_root)
    atlas.local_version = (1, 3)
    second = atlas_cache.atlas_cache_dir(atlas, cache_root)
    atlas.resolution = (50, 50, 50)
    third = atlas_cache.atlas_cache_dir(atlas, cache_root)

    assert first == (
        cache_root
        / "atlases"
        / "v1"
        / "example_mouse_25um"
        / "1.2"
        / "25x25x25um"
    )
    assert len({first, second, third}) == 3


def test_reference_derivative_uses_app_cache_and_is_reused(
    tmp_path: Path,
) -> None:
    atlas_root = tmp_path / "brainglobe" / "example_v1.2"
    atlas_root.mkdir(parents=True)
    sentinel = atlas_root / "source-atlas-file"
    sentinel.write_text("unchanged", encoding="utf-8")
    atlas = _fake_atlas(atlas_root)
    cache_root = tmp_path / "app-cache"

    converted = atlas_cache.load_reference_8bit(atlas, cache_root)
    atlas.template.fill(0)
    cached = atlas_cache.load_reference_8bit(atlas, cache_root)

    assert converted.dtype == np.dtype(np.uint8)
    assert converted.shape == atlas.shape
    assert converted.min() == 0
    assert converted.max() == 255
    np.testing.assert_array_equal(cached, converted)
    assert (
        atlas_cache.atlas_cache_dir(atlas, cache_root)
        / "reference_8bit.npy"
    ).exists()
    assert list(atlas_root.iterdir()) == [sentinel]


def test_incompatible_reference_cache_is_regenerated(tmp_path: Path) -> None:
    atlas = _fake_atlas(tmp_path / "brainglobe" / "example_v1.2")
    cache_root = tmp_path / "app-cache"
    cache_path = (
        atlas_cache.atlas_cache_dir(atlas, cache_root)
        / "reference_8bit.npy"
    )
    np.save(cache_path, np.zeros((1,), dtype=np.uint8))

    regenerated = atlas_cache.load_reference_8bit(atlas, cache_root)

    assert regenerated.shape == atlas.shape
    assert regenerated.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(np.load(cache_path), regenerated)


def test_reference_loader_prefers_v3_template_attribute(tmp_path: Path) -> None:
    class TemplateOnlyAtlas:
        atlas_name = "future_atlas_25um"
        local_version = (3, 0)
        metadata = {"name": "future_atlas"}
        resolution = (25, 25, 25)
        template = np.arange(8, dtype=np.uint16).reshape((2, 2, 2))
        shape = template.shape

        @property
        def reference(self):
            raise AssertionError("the deprecated reference alias was accessed")

    converted = atlas_cache.load_reference_8bit(
        TemplateOnlyAtlas(),
        tmp_path / "app-cache",
    )

    assert converted.dtype == np.dtype(np.uint8)
    assert converted[0, 0, 0] == 0
    assert converted[-1, -1, -1] == 255


def test_load_annot_bool_never_writes_to_brainglobe_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    atlas_root = tmp_path / "brainglobe" / "example_v1.2"
    atlas_root.mkdir(parents=True)
    sentinel = atlas_root / "source-atlas-file"
    sentinel.write_text("unchanged", encoding="utf-8")
    atlas = _fake_atlas(atlas_root)
    app_cache = tmp_path / "app-cache"
    monkeypatch.setattr(
        atlas_cache,
        "user_cache_path",
        lambda *args, **kwargs: app_cache,
    )
    monkeypatch.setattr(atlas_utils, "BrainGlobeAtlas", lambda name: atlas)

    annotation_bool = atlas_utils.loadAnnotBool("example_mouse_25um")

    np.testing.assert_array_equal(
        annotation_bool,
        (atlas.annotation > 0).astype(np.uint8) * 255,
    )
    assert annotation_bool.dtype == np.dtype(np.uint8)
    assert (
        atlas_cache.atlas_cache_dir(atlas, app_cache) / "annot_bool.npy"
    ).exists()
    assert list(atlas_root.iterdir()) == [sentinel]


def test_registration_template_loader_uses_shared_cache(
    monkeypatch,
) -> None:
    model = object.__new__(AtlasModel)
    model.atlas = object()
    model.regi_dict = {"orientation": "horizontal"}
    template = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    monkeypatch.setattr(
        "napari_dmc_brainmap.registration.sharpy_track.sharpy_track."
        "model.AtlasModel.load_reference_8bit",
        lambda atlas: template,
    )

    model.loadTemplate()

    np.testing.assert_array_equal(model.template, template.transpose(1, 0, 2))


def test_probe_template_loader_uses_shared_cache(monkeypatch) -> None:
    visualizer = ProbeVisualizer.__new__(ProbeVisualizer)
    visualizer.atlas = object()
    template = np.arange(8, dtype=np.uint8).reshape((2, 2, 2))
    monkeypatch.setattr(
        "napari_dmc_brainmap.results.probe_vis.probe_vis.view."
        "ProbeVisualizer.load_reference_8bit",
        lambda atlas: template,
    )

    visualizer.loadTemplate()

    np.testing.assert_array_equal(visualizer.template, template)
