"""
CMake integration for TrueForm C++ library

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""


def get_include() -> str:
    """Return the path to TrueForm C++ headers."""
    from pathlib import Path
    return str(Path(__file__).parent.parent / "include")


def get_cmake_dir() -> str:
    """Return the path to TrueForm CMake configuration."""
    from pathlib import Path
    return str(Path(__file__).parent.parent / "cmake_config")


__all__ = ["get_include", "get_cmake_dir"]
