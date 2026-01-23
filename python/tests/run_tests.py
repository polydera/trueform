#!/usr/bin/env python3
"""
Test runner for trueform Python bindings

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py test_file.py       # Run specific test file

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import os
import glob
import subprocess


def run_test_file(test_file):
    """Run a single test file as a subprocess"""
    print(f"\n{'=' * 70}")
    print(f"Running: {os.path.basename(test_file)}")
    print('=' * 70)

    # Run the test file as a standalone Python script
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True
    )

    # Print stdout
    if result.stdout:
        print(result.stdout, end='')

    # Print stderr
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)

    # Check return code
    if result.returncode == 0:
        return True
    else:
        print(f"\n[FAIL] Test failed with exit code: {result.returncode}")
        return False


def main():
    # Get the directory where this script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        # Run specific test file
        test_file_arg = sys.argv[1]
        if not os.path.isabs(test_file_arg):
            test_file_arg = os.path.join(test_dir, test_file_arg)
        test_files = [test_file_arg]
    else:
        # Run all test files
        test_files = glob.glob(os.path.join(test_dir, "test_*.py"))

    if not test_files:
        print("No test files found!")
        return 1

    print(f"Found {len(test_files)} test file(s)")

    passed = 0
    failed = 0
    failed_tests = []

    for test_file in sorted(test_files):
        if run_test_file(test_file):
            passed += 1
        else:
            failed += 1
            failed_tests.append(os.path.basename(test_file))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total:  {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")

    if failed == 0:
        print("\n[PASS] ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n[FAIL] {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
