#!/usr/bin/env python3
"""
Full verification: build, install, and test trueform.

This script runs:
1. build.py - clone, build, install, verify
2. tests.py - run C++ and Python tests
3. Cleanup (unless --keep is specified)

Usage:
    python verify/all.py [options]

Examples:
    python verify/all.py                       # Full verification
    python verify/all.py --work-dir ./build    # Use specific work directory
    python verify/all.py --skip-vtk            # Skip VTK integration
    python verify/all.py --skip-python         # Skip Python package
    python verify/all.py --keep                # Keep build artifacts
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# Import the run functions from sibling modules
from build import run_build, colored, Colors
from tests import run_tests


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

    args = parser.parse_args()

    # Set default work_dir if not specified
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = Path(tempfile.gettempdir()) / "trueform-verify"

    # Run build (always keep artifacts for tests)
    build_success = run_build(
        work_dir=work_dir,
        install_prefix=args.install_prefix,
        skip_vtk=args.skip_vtk,
        skip_python=args.skip_python,
        branch=args.branch,
        keep=True,  # Always keep for tests
    )

    if not build_success:
        # Cleanup on failure unless --keep
        if not args.keep and work_dir.exists():
            print(f"\nCleaning up {work_dir}...")
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                print(colored(f"Warning: Could not clean up: {e}", Colors.YELLOW))
        return 1

    # Run tests using the work directory
    test_success = run_tests(
        work_dir=work_dir,
        skip_cpp=False,
        skip_python=args.skip_python,
    )

    # Cleanup unless --keep
    if not args.keep and work_dir.exists():
        print(f"\nCleaning up {work_dir}...")
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            print(colored(f"Warning: Could not clean up: {e}", Colors.YELLOW))

    return 0 if (build_success and test_success) else 1


if __name__ == "__main__":
    sys.exit(main())
