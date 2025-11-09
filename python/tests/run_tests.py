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
import importlib.util

# Add parent directory to path so we can import trueform
# This assumes the test runner is in build/python/tests/ and trueform is in build/python/trueform/
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_test_file(test_file):
    """Run a single test file"""
    print(f"\n{'=' * 70}")
    print(f"Running: {test_file}")
    print('=' * 70)

    # Load and execute the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Get the directory where this script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        # Run specific test file
        test_files = [os.path.join(test_dir, sys.argv[1])]
    else:
        # Run all test files
        test_files = glob.glob(os.path.join(test_dir, "test_*.py"))

    if not test_files:
        print("No test files found!")
        return 1

    print(f"Found {len(test_files)} test file(s)")

    passed = 0
    failed = 0

    for test_file in sorted(test_files):
        if run_test_file(test_file):
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total:  {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
