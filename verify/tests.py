#!/usr/bin/env python3
"""
Run trueform C++ and Python tests.

Usage:
    python verify/tests.py [options]

Examples:
    python verify/tests.py                           # Run tests (builds if needed)
    python verify/tests.py --work-dir ./my-build     # Use specific work directory
    python verify/tests.py --skip-cpp                # Only Python tests
    python verify/tests.py --skip-python             # Only C++ tests
"""

import argparse
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from venv_utils import VenvInfo, get_venv_info


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


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
    """Run Python tests via run_tests.py."""
    import re

    print_step("Python Tests")

    test_runner = source_dir / "python" / "tests" / "run_tests.py"
    if not test_runner.exists():
        print_fail("Python tests", f"Test runner not found: {test_runner}")
        return False

    # Install pytest and run tests
    output = ""
    success = False

    if venv_info:
        # Install pytest in venv
        try:
            venv_info.run_pip(["install", "pytest"])
            print_pass("Install pytest")
        except subprocess.CalledProcessError as e:
            print_fail("Install pytest", getattr(e, 'stdout', str(e)))
            return False

        # Run tests using venv
        try:
            result = venv_info.run_python([str(test_runner)], cwd=source_dir)
            output = result.stdout
            success = True
        except subprocess.CalledProcessError as e:
            output = getattr(e, 'stdout', str(e))
    else:
        # Run tests using current Python
        try:
            result = run_cmd([sys.executable, str(test_runner)], cwd=source_dir, capture=True)
            output = result.stdout
            success = True
        except subprocess.CalledProcessError as e:
            output = getattr(e, 'stdout', str(e))

    # Parse output - format is "Total: X\nPassed: Y\nFailed: Z"
    total_match = re.search(r'Total:\s*(\d+)', output)
    passed_match = re.search(r'Passed:\s*(\d+)', output)
    failed_match = re.search(r'Failed:\s*(\d+)', output)

    if passed_match and total_match:
        passed = passed_match.group(1)
        total = total_match.group(1)
        failed = failed_match.group(1) if failed_match else "0"
        if success and ("ALL TESTS PASSED" in output or failed == "0"):
            print_pass(f"pytest ({passed}/{total} passed)")
            return True
        else:
            print_fail("pytest", f"{passed}/{total} passed, {failed} failed")
            return False

    if success and "ALL TESTS PASSED" in output:
        print_pass("pytest")
        return True

    print_fail("pytest", "Tests failed")
    return False


def build_for_tests(work_dir: Path, source_dir: Path) -> Path:
    """Build the project for testing."""
    clone_dir = work_dir / "trueform"
    build_dir = clone_dir / "build"

    num_jobs = max(1, multiprocessing.cpu_count())

    print_step("Setup")

    if work_dir.exists():
        try:
            shutil.rmtree(work_dir)
            print_pass("Clean work directory")
        except Exception as e:
            print_fail("Clean work directory", str(e))
            return None

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        print_pass("Create work directory")
    except Exception as e:
        print_fail("Create work directory", str(e))
        return None

    print_step("Clone Repository")

    try:
        run_cmd(["git", "clone", str(source_dir), str(clone_dir)], cwd=work_dir, capture=True)
        print_pass("Clone repository")
    except subprocess.CalledProcessError as e:
        print_fail("Clone repository", getattr(e, 'stdout', str(e)))
        return None

    print_step("Configure")

    try:
        run_cmd([
            "cmake",
            "-S", str(clone_dir),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTF_BUILD_TESTS=ON",
        ], capture=True)
        print_pass("Configure CMake")
    except subprocess.CalledProcessError as e:
        print_fail("Configure CMake", getattr(e, 'stdout', str(e)))
        return None

    print_step("Build Tests")

    try:
        run_cmd([
            "cmake", "--build", str(build_dir),
            "--target", "trueform_tests",
            "--parallel", str(num_jobs),
        ], capture=True)
        print_pass(f"Build trueform_tests (jobs={num_jobs})")
    except subprocess.CalledProcessError as e:
        print_fail("Build trueform_tests", getattr(e, 'stdout', str(e)))
        return None

    return build_dir


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
        work_dir = Path(tempfile.gettempdir()) / "trueform-verify"

    # Derive build_dir from work_dir
    build_dir = work_dir / "trueform" / "build"

    print(colored("Trueform Tests", Colors.BOLD + Colors.BLUE))
    print(f"  Source: {source_dir}")
    print(f"  Work:   {work_dir}")

    cpp_passed = True
    python_passed = True

    # Check if build exists, otherwise build it
    venv_info = None
    if build_dir.exists():
        print(f"  Build:  {build_dir}")
        clone_dir = work_dir / "trueform"
        # Check for venv
        venv_info = get_venv_info(work_dir / "venv")
        if venv_info:
            print(f"  Venv:   {venv_info.venv_dir}")
    else:
        print("  Build:  (will build)")
        build_dir = build_for_tests(work_dir, source_dir)
        if build_dir is None:
            return False
        clone_dir = work_dir / "trueform"

    if skip_cpp:
        print_step("C++ Tests")
        print_skip("ctest", "skipped by user")
    else:
        cpp_passed = run_cpp_tests(build_dir)

    if skip_python:
        print_step("Python Tests")
        print_skip("pytest", "skipped by user")
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
