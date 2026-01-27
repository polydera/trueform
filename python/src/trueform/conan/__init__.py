"""
Conan integration for TrueForm C++ library

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from pathlib import Path


def get_conan_dir() -> str:
    """Return the path to directory containing conanfile.py."""
    return str(Path(__file__).parent)


__all__ = ["get_conan_dir"]
