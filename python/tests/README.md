# Trueform Python Bindings Tests

This directory contains tests for the trueform Python bindings.

## Running Tests

### From the build directory

After building the Python bindings:

```bash
cd build/python
source venv/bin/activate  # if using a virtual environment
python tests/run_tests.py
```

### Run all tests

```bash
python tests/run_tests.py
```

### Run a specific test file

```bash
python tests/test_closest_metric_point_pair.py
# or
python tests/run_tests.py test_closest_metric_point_pair.py
```

## Test Structure

Tests should follow the naming convention `test_*.py` and will be automatically discovered by the test runner.

Each test file should:
1. Import `trueform as tf`
2. Define test functions with descriptive names
3. Use assertions to verify correctness
4. Print clear success/failure messages
5. Exit with code 1 on failure (for CI integration)
