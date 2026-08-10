import platform
import sys


def ensure_supported_platform(
    sys_platform: str | None = None,
    machine: str | None = None,
) -> None:
    """Reject Intel macOS with a clear message during plugin discovery."""
    current_platform = sys.platform if sys_platform is None else sys_platform
    current_machine = platform.machine() if machine is None else machine

    if current_platform == "darwin" and current_machine.lower() in {
        "amd64",
        "x86_64",
    }:
        raise RuntimeError(
            "DMC-BrainMap supports macOS only on Apple Silicon; "
            "Intel Macs are not supported."
        )
