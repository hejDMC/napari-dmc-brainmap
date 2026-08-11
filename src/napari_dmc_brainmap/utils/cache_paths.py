"""Shared application-cache locations."""

from pathlib import Path

from platformdirs import user_cache_path


APP_NAME = "napari-dmc-brainmap"


def app_cache_root() -> Path:
    """Return the application-owned platform cache directory."""
    cache_root = Path(user_cache_path(APP_NAME, appauthor=False))
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root
