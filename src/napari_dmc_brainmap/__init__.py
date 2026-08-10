from importlib.metadata import PackageNotFoundError, version

from napari_dmc_brainmap._platform import ensure_supported_platform


ensure_supported_platform()

try:
    __version__ = version("napari-dmc-brainmap")
except PackageNotFoundError:
    __version__ = "0+unknown"
