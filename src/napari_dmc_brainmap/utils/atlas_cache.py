"""Application-owned cache for arrays derived from BrainGlobe atlases."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from platformdirs import user_cache_path


APP_NAME = "napari-dmc-brainmap"
CACHE_SCHEMA_VERSION = "v1"


def _path_component(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return component or fallback


def _atlas_name(atlas: Any) -> str:
    metadata = getattr(atlas, "metadata", {}) or {}
    name = getattr(atlas, "atlas_name", None) or metadata.get("name")
    return _path_component(name, "unknown-atlas")


def _atlas_version(atlas: Any) -> str:
    metadata = getattr(atlas, "metadata", {}) or {}
    version = metadata.get("version") or metadata.get("atlas_version")
    if version is None:
        version = getattr(atlas, "local_version", None)
    if version is None:
        root_name = Path(getattr(atlas, "root_dir", "")).name
        if "_v" in root_name:
            version = root_name.rsplit("_v", maxsplit=1)[-1]
    if isinstance(version, (tuple, list)):
        version = ".".join(str(part) for part in version)
    return _path_component(version, "unknown-version")


def _atlas_resolution(atlas: Any) -> str:
    resolution = getattr(atlas, "resolution", ())
    if not resolution:
        metadata = getattr(atlas, "metadata", {}) or {}
        resolution = metadata.get("resolution", ())
    if not resolution:
        return "unknown-resolution"
    values = "x".join(f"{float(value):g}" for value in resolution)
    return _path_component(f"{values}um", "unknown-resolution")


def atlas_cache_dir(
    atlas: Any,
    cache_root: Path | None = None,
) -> Path:
    """Return the versioned cache directory for a specific atlas."""
    root = cache_root or user_cache_path(APP_NAME, appauthor=False)
    cache_dir = (
        Path(root)
        / "atlases"
        / CACHE_SCHEMA_VERSION
        / _atlas_name(atlas)
        / _atlas_version(atlas)
        / _atlas_resolution(atlas)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_valid_array(
    cache_path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
    except (EOFError, OSError, ValueError):
        return None
    if cached.shape != expected_shape or cached.dtype != expected_dtype:
        return None
    return cached


def _save_array(cache_path: Path, array: np.ndarray) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".npy",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, array, allow_pickle=False)
        temporary_path.replace(cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atlas_template(atlas: Any) -> np.ndarray:
    """Return the v3 template or the v1/v2 reference fallback."""
    template = getattr(atlas, "template", None)
    if template is None:
        template = atlas.reference
    return np.asarray(template)


def load_reference_8bit(
    atlas: Any,
    cache_root: Path | None = None,
    notify: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Load or create the plugin's uint8 atlas-template derivative."""
    expected_shape = tuple(int(value) for value in atlas.shape)
    cache_path = atlas_cache_dir(atlas, cache_root) / "reference_8bit.npy"
    cached = _load_valid_array(cache_path, expected_shape, np.dtype(np.uint8))
    if cached is not None:
        if notify is not None:
            notify("loading template volume...")
        return cached

    template = _atlas_template(atlas)
    if template.dtype == np.dtype(np.uint8):
        return template
    if template.dtype != np.dtype(np.uint16):
        return template

    if notify is not None:
        notify("creating 8-bit template volume...")
    minimum = template.min()
    maximum = template.max()
    if maximum == minimum:
        converted = np.zeros(template.shape, dtype=np.uint8)
    else:
        converted = (
            (template.astype(np.float64) - minimum)
            / (maximum - minimum)
            * 255
        ).astype(np.uint8)
    _save_array(cache_path, converted)
    return converted


def load_annotation_bool(
    atlas: Any,
    cache_root: Path | None = None,
    notify: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Load or create the plugin's uint8 binary annotation derivative."""
    expected_shape = tuple(int(value) for value in atlas.shape)
    cache_path = atlas_cache_dir(atlas, cache_root) / "annot_bool.npy"
    cached = _load_valid_array(cache_path, expected_shape, np.dtype(np.uint8))
    if cached is not None:
        if notify is not None:
            notify("loading annot_bool volume...")
        return cached

    if notify is not None:
        notify("... local version not found, loading annotation volume...")
    annotation = np.asarray(atlas.annotation)
    if notify is not None:
        notify("... creating annot_bool version...")
    annotation_bool = (annotation > 0).astype(np.uint8) * 255
    _save_array(cache_path, annotation_bool)
    return annotation_bool
