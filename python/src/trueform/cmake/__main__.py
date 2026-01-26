"""
CMake integration CLI for TrueForm

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import argparse
import sys

from . import get_include, get_cmake_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrueForm CMake integration helpers"
    )
    parser.add_argument(
        "--include_dir",
        action="store_true",
        help="Print the path to TrueForm C++ headers"
    )
    parser.add_argument(
        "--cmake_dir",
        action="store_true",
        help="Print the path to TrueForm CMake configuration"
    )

    args = parser.parse_args()

    # Default: print cmake_dir (most common use case)
    if not sys.argv[1:]:
        print(get_cmake_dir())
        return

    if args.include_dir:
        print(get_include())
    if args.cmake_dir:
        print(get_cmake_dir())


if __name__ == "__main__":
    main()
