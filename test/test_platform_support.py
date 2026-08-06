import pytest

from napari_dmc_brainmap._platform import ensure_supported_platform


@pytest.mark.parametrize(
    ("sys_platform", "machine"),
    [
        ("darwin", "arm64"),
        ("linux", "x86_64"),
        ("win32", "AMD64"),
    ],
)
def test_supported_platforms_are_accepted(sys_platform, machine):
    ensure_supported_platform(sys_platform, machine)


@pytest.mark.parametrize("machine", ["x86_64", "AMD64"])
def test_intel_macos_is_rejected_with_clear_message(machine):
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        ensure_supported_platform("darwin", machine)
