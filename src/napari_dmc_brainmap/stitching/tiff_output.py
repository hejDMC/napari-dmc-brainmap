"""Completion-safe TIFF output helpers for stitched images."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import disk_usage
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import numpy as np
import tifffile

from napari_dmc_brainmap.utils.cache_paths import app_cache_root

DEFLATE_LEVEL = 1
_BIGTIFF_THRESHOLD = 2**32 - 2**25
_MINIMUM_SCRATCH_MARGIN_BYTES = 64 * 1024**2


def stitching_scratch_root() -> Path:
    """Return the application-owned directory for transient stitch canvases."""
    scratch_root = app_cache_root() / "stitching" / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    return scratch_root


def _format_bytes(size: int) -> str:
    """Return a concise binary size for user-facing messages."""
    value = float(size)
    for unit in ("bytes", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _check_scratch_space(scratch_root: Path, canvas_bytes: int) -> None:
    """Fail before allocation when the local app cache lacks working space."""
    safety_margin = max(
        _MINIMUM_SCRATCH_MARGIN_BYTES,
        canvas_bytes // 10,
    )
    required_bytes = canvas_bytes + safety_margin
    available_bytes = disk_usage(scratch_root).free
    if available_bytes >= required_bytes:
        return

    message = (
        "Not enough local application-cache space to stitch this image. "
        f"Cache location: {scratch_root}. "
        f"Required: {_format_bytes(required_bytes)} "
        f"({_format_bytes(canvas_bytes)} image plus safety margin). "
        f"Available: {_format_bytes(available_bytes)}. "
        "Free space on this drive, then run stitching again. "
        "No raw or destination image files were modified."
    )
    print(message)
    raise OSError(message)


def write_tiff_atomically(
    destination: str | Path,
    image: np.ndarray,
    **write_options: Any,
) -> None:
    """Write a TIFF beside its destination and publish it after completion."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as partial_file:
            partial_path = Path(partial_file.name)
        tifffile.imwrite(partial_path, image, **write_options)
        with partial_path.open("rb+") as completed_file:
            completed_file.flush()
            os.fsync(completed_file.fileno())
        partial_path.replace(destination)
        partial_path = None
    finally:
        if partial_path is not None:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(
                    "Could not remove the incomplete TIFF created by this "
                    f"operation: {partial_path}. Error: {cleanup_error}"
                )


def _write_compressed_stitched_tiff(
    destination: Path,
    canvas: np.ndarray,
) -> None:
    """Stream a memmapped canvas into a losslessly compressed TIFF."""
    print(f"Saving stitched image: {destination}")
    write_tiff_atomically(
        destination,
        canvas,
        photometric="minisblack",
        compression="deflate",
        compressionargs={"level": DEFLATE_LEVEL},
        predictor=True,
        bigtiff=canvas.nbytes >= _BIGTIFF_THRESHOLD,
        metadata=None,
    )


@contextmanager
def compressed_stitch_canvas(
    destination: str | Path,
    shape: tuple[int, int],
) -> Iterator[np.memmap]:
    """Yield an uncompressed scratch canvas and publish a compressed TIFF.

    Scratch data always lives in the application's platform cache. The
    destination is replaced only after compression finishes successfully.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = stitching_scratch_root()
    canvas_bytes = (
        int(np.prod(shape, dtype=np.int64))
        * np.dtype(np.uint16).itemsize
    )
    _check_scratch_space(scratch_root, canvas_bytes)

    with TemporaryDirectory(
        prefix=f".{destination.stem}-stitching-",
        dir=scratch_root,
        ignore_cleanup_errors=True,
    ) as temporary_directory:
        scratch_path = Path(temporary_directory) / "canvas.tif"
        canvas = tifffile.memmap(
            scratch_path,
            shape=shape,
            dtype=np.uint16,
            photometric="minisblack",
        )
        try:
            yield canvas
            canvas.flush()
            _write_compressed_stitched_tiff(destination, canvas)
        finally:
            canvas.flush()
            memory_map = getattr(canvas, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
