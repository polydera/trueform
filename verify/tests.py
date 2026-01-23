#!/usr/bin/env python3
"""
Run trueform C++ and Python tests.

This script runs tests on existing builds. It does not build anything.
If the build or venv is missing, tests are skipped.

Note: Use `python -m verify` to build and run tests together.

Usage:
    python -m verify.tests [options]

Examples:
    python -m verify.tests                           # Run all tests
    python -m verify.tests --work-dir ./my-build     # Use specific work directory
    python -m verify.tests --skip-cpp                # Only Python tests
    python -m verify.tests --skip-python             # Only C++ tests
"""

import argparse
import multiprocessing
import subprocess
import sys
from pathlib import Path

from .build import get_default_work_dir, Colors, colored
from .venv_utils import VenvInfo, get_venv_info


def print_header(name: str) -> None:
    width = 50
    print()
    print(colored("=" * width, Colors.BOLD))
    print(colored(f"  {name}", Colors.BOLD))
    print(colored("=" * width, Colors.BOLD))


def print_step(name: str) -> None:
    print(f"\n{colored('==>', Colors.BLUE)} {colored(name, Colors.BOLD)}")


def print_pass(name: str) -> None:
    print(f"  {colored('[PASS]', Colors.GREEN)} {name}")


def print_fail(name: str, error: str = "") -> None:
    print(f"  {colored('[FAIL]', Colors.RED)} {name}")
    if error:
        for line in error.strip().split("\n")[:10]:
            print(f"         {line}")


def print_skip(name: str, reason: str = "") -> None:
    msg = f"  {colored('[SKIP]', Colors.YELLOW)} {name}"
    if reason:
        msg += f" ({reason})"
    print(msg)


def run_cmd(cmd: list, cwd: Path = None, capture: bool = False):
    """Run command."""
    kwargs = {"cwd": cwd, "check": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def run_cpp_tests(build_dir: Path) -> bool:
    """Run C++ tests via ctest."""
    print_step("C++ Tests")

    num_jobs = max(1, multiprocessing.cpu_count())

    try:
        result = run_cmd(
            [
                "ctest",
                "--test-dir", str(build_dir / "tests"),
                "--output-on-failure",
                "-j", str(num_jobs),
            ],
            cwd=build_dir,
            capture=True,
        )
        # Parse ctest output for summary
        output = result.stdout
        # Look for line like "100% tests passed, 0 tests failed out of 1013"
        import re
        for line in output.split("\n"):
            match = re.search(r'(\d+)%\s+tests\s+passed.*?(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)', line, re.IGNORECASE)
            if match:
                pct, failed, total = match.groups()
                passed = int(total) - int(failed)
                print_pass(f"ctest ({passed}/{total} passed)")
                return True
        print_pass("ctest")
        return True
    except subprocess.CalledProcessError as e:
        output = getattr(e, 'stdout', '') or str(e)
        # Try to extract failure info
        fail_info = ""
        failed_tests = []
        import re

        # Look for "The following tests FAILED:" section
        in_failed_section = False
        for line in output.split("\n"):
            if "The following tests FAILED:" in line:
                in_failed_section = True
                continue
            if in_failed_section:
                # Lines look like: "  123 - core::test_name (Failed)"
                test_match = re.search(r'^\s*\d+\s+-\s+(.+?)\s+\(', line)
                if test_match:
                    failed_tests.append(test_match.group(1))
                elif line.strip() and not line.startswith(" "):
                    in_failed_section = False

            # Look for summary line
            match = re.search(r'(\d+)%\s+tests\s+passed.*?(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)', line, re.IGNORECASE)
            if match:
                pct, failed, total = match.groups()
                passed = int(total) - int(failed)
                fail_info = f"{passed}/{total} passed, {failed} failed"

        # Build error message
        if failed_tests:
            fail_info = f"{fail_info}\n         Failed: {', '.join(failed_tests)}"

        print_fail("ctest", fail_info if fail_info else "Tests failed")
        return False


def run_python_tests(source_dir: Path, venv_info: VenvInfo = None) -> bool:
    """Run Python tests via pytest."""
    import re

    print_step("Python Tests")

    test_dir = source_dir / "python" / "tests"
    if not test_dir.exists():
        print_fail("Python tests", f"Test directory not found: {test_dir}")
        return False

    output = ""
    success = False

    if venv_info:
        try:
            venv_info.run_pip(["install", "pytest"])
            print_pass("Install pytest")
        except subprocess.CalledProcessError as e:
            print_fail("Install pytest", getattr(e, 'stdout', str(e)))
            return False

        try:
            result = venv_info.run_python(["-m", "pytest", str(test_dir), "-v"], cwd=source_dir)
            output = result.stdout
            success = True
        except subprocess.CalledProcessError as e:
            output = getattr(e, 'stdout', str(e))
    else:
        try:
            result = run_cmd([sys.executable, "-m", "pytest", str(test_dir), "-v"], cwd=source_dir, capture=True)
            output = result.stdout
            success = True
        except subprocess.CalledProcessError as e:
            output = getattr(e, 'stdout', str(e))

    # Parse pytest summary line: "X passed" or "X passed, Y failed" or "X errors"
    match = re.search(r'(\d+)\s+passed', output)
    failed_match = re.search(r'(\d+)\s+failed', output)
    error_match = re.search(r'(\d+)\s+errors?', output)

    passed = int(match.group(1)) if match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(error_match.group(1)) if error_match else 0
    total = passed + failed

    if errors > 0:
        print_fail("pytest", f"{errors} collection errors")
        return False

    if total > 0:
        if success and failed == 0:
            print_pass(f"pytest ({passed}/{total} passed)")
            return True
        else:
            print_fail("pytest", f"{passed}/{total} passed, {failed} failed")
            return False

    if success:
        print_pass("pytest")
        return True

    print_fail("pytest", "Tests failed")
    return False


def run_tests(
    work_dir: Path = None,
    skip_cpp: bool = False,
    skip_python: bool = False,
    source_dir: Path = None,
) -> bool:
    """
    Run trueform C++ and Python tests.

    Args:
        work_dir: Working directory (default: <tempdir>/trueform-verify)
        skip_cpp: Skip C++ tests
        skip_python: Skip Python tests
        source_dir: Source directory (defaults to parent of verify/)

    Returns True if all tests passed, False otherwise.
    """
    if source_dir is None:
        source_dir = Path(__file__).resolve().parent.parent
    if work_dir is None:
        work_dir = get_default_work_dir()

    clone_dir = work_dir / "trueform"
    build_dir = clone_dir / "build"
    venv_info = get_venv_info(work_dir / "venv")

    print(colored("Trueform Tests", Colors.BOLD + Colors.BLUE))
    print(f"  Source: {source_dir}")
    print(f"  Work:   {work_dir}")
    if build_dir.exists():
        print(f"  Build:  {build_dir}")
    if venv_info:
        print(f"  Venv:   {venv_info.venv_dir}")

    cpp_passed = True
    python_passed = True

    # C++ tests
    if skip_cpp:
        print_step("C++ Tests")
        print_skip("ctest", "skipped by user")
    elif not build_dir.exists():
        print_step("C++ Tests")
        print_skip("ctest", "no build found")
    else:
        cpp_passed = run_cpp_tests(build_dir)

    # Python tests
    if skip_python:
        print_step("Python Tests")
        print_skip("pytest", "skipped by user")
    elif not venv_info:
        print_step("Python Tests")
        print_skip("pytest", "no venv found")
    else:
        python_passed = run_python_tests(clone_dir, venv_info)

    print_step("Summary")

    all_passed = cpp_passed and python_passed

    if not skip_cpp:
        if cpp_passed:
            print(f"  {colored('[PASS]', Colors.GREEN)} C++ Tests")
        else:
            print(f"  {colored('[FAIL]', Colors.RED)} C++ Tests")

    if not skip_python:
        if python_passed:
            print(f"  {colored('[PASS]', Colors.GREEN)} Python Tests")
        else:
            print(f"  {colored('[FAIL]', Colors.RED)} Python Tests")

    print()
    if all_passed:
        print(colored("All tests passed!", Colors.GREEN + Colors.BOLD))
    else:
        print(colored("Some tests failed!", Colors.RED + Colors.BOLD))

    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run trueform C++ and Python tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory (default: <tempdir>/trueform-verify)",
    )
    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip C++ tests",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python tests",
    )
    args = parser.parse_args()

    success = run_tests(
        work_dir=args.work_dir,
        skip_cpp=args.skip_cpp,
        skip_python=args.skip_python,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
