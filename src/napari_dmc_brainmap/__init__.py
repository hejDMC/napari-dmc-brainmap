from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("napari-dmc-brainmap")
except PackageNotFoundError:
    __version__ = "0+unknown"
