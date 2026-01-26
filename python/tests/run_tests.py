#!/usr/bin/env python3
"""
Test runner for trueform Python bindings

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py test_file.py       # Run specific test file

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import sys
import os
import glob
import subprocess


def run_test_file(test_file):
    """Run a single test file as a subprocess"""
    # Run the test file directly - it calls pytest internally via __main__
    result = subprocess.run(
        [sys.executable, test_file],
    )
    return result.returncode == 0


def main():
    test_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        # Run specific test file(s)
        test_files = []
        for arg in sys.argv[1:]:
            if not os.path.isabs(arg):
                arg = os.path.join(test_dir, arg)
            test_files.append(arg)
    else:
        # Run all test files
        test_files = sorted(glob.glob(os.path.join(test_dir, "test_*.py")))

    if not test_files:
        print("No test files found!")
        return 1

    print(f"Running {len(test_files)} test file(s)\n")

    passed = 0
    failed = 0
    failed_tests = []

    for test_file in test_files:
        if run_test_file(test_file):
            passed += 1
        else:
            failed += 1
            failed_tests.append(os.path.basename(test_file))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test files: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed_tests:
        print(f"\nFailed: {', '.join(failed_tests)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
