#!/usr/bin/env python3
"""
Trueform verification script.

Modes:
    (default)       Full verification: build, install, verify, and test everything
    --cpp-only      C++ only: build and test C++ (examples, tests, VTK)
    --python-only   Python only: build bindings, install, verify, and run pytest

Usage:
    python -m verify [options]

Examples:
    python -m verify                                        # Full verification
    python -m verify --cpp-only                             # C++ build and tests
    python -m verify --cpp-only --skip-vtk --skip-examples  # Fast C++ tests only
    python -m verify --python-only                          # Python tests only
    python -m verify --keep                                 # Keep build artifacts
"""

import argparse
import sys
from pathlib import Path

from .build import run_build, run_build_cpp_only, run_build_python_only, colored, Colors, get_default_work_dir, robust_rmtree
from .tests import run_tests


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full verification: build, install, and test trueform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory (default: <tempdir>/trueform-verify)",
    )
    parser.add_argument(
        "--install-prefix",
        type=Path,
        default=None,
        help="Installation prefix (default: <work-dir>/install)",
    )
    parser.add_argument(
        "--skip-vtk",
        action="store_true",
        help="Skip VTK integration",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python package",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Skip building examples",
    )
    parser.add_argument(
        "--cpp-only",
        action="store_true",
        help="C++ build and tests only",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Python build and tests only",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to clone",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep build artifacts",
    )
    parser.add_argument(
        "--toolchain-file",
        type=Path,
        default=None,
        help="CMake toolchain file (e.g., vcpkg.cmake)",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.cpp_only and args.python_only:
        print("Error: --cpp-only and --python-only are mutually exclusive")
        return 1

    # Set default work_dir if not specified
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = get_default_work_dir()

    # Determine mode
    if args.cpp_only:
        build_success = run_build_cpp_only(
            work_dir=work_dir,
            skip_vtk=args.skip_vtk,
            skip_examples=args.skip_examples,
            branch=args.branch,
            keep=True,
            toolchain_file=args.toolchain_file,
        )
        if not build_success:
            if not args.keep and work_dir.exists():
                _cleanup(work_dir)
            return 1

        test_success = run_tests(
            work_dir=work_dir,
            skip_cpp=False,
            skip_python=True,
        )

    elif args.python_only:
        build_success = run_build_python_only(
            work_dir=work_dir,
            branch=args.branch,
            keep=True,
            toolchain_file=args.toolchain_file,
        )
        if not build_success:
            if not args.keep and work_dir.exists():
                _cleanup(work_dir)
            return 1

        test_success = run_tests(
            work_dir=work_dir,
            skip_cpp=True,
            skip_python=False,
        )

    else:
        # Full verification
        build_success = run_build(
            work_dir=work_dir,
            install_prefix=args.install_prefix,
            skip_vtk=args.skip_vtk,
            skip_python=args.skip_python,
            skip_examples=args.skip_examples,
            branch=args.branch,
            keep=True,
            toolchain_file=args.toolchain_file,
        )
        if not build_success:
            if not args.keep and work_dir.exists():
                _cleanup(work_dir)
            return 1

        test_success = run_tests(
            work_dir=work_dir,
            skip_cpp=False,
            skip_python=args.skip_python,
        )

    # Cleanup unless --keep
    if not args.keep and work_dir.exists():
        _cleanup(work_dir)

    return 0 if (build_success and test_success) else 1


def _cleanup(work_dir: Path) -> None:
    print(f"\nCleaning up {work_dir}...")
    try:
        robust_rmtree(work_dir)
    except Exception as e:
        print(colored(f"Warning: Could not clean up: {e}", Colors.YELLOW))


if __name__ == "__main__":
    sys.exit(main())
