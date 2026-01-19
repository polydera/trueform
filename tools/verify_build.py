#!/usr/bin/env python3
"""
Verify trueform builds correctly from a clean git clone.

This script tests that:
1. The repository can be cloned and built
2. Core library, examples, and VTK integration build successfully
3. Installation creates the expected directory structure
4. find_package works from an external project
5. Python package installs via pip and works correctly

Usage:
    python tools/verify_build.py [options]

Examples:
    # Full verification (all components)
    python tools/verify_build.py

    # Skip VTK integration tests
    python tools/verify_build.py --skip-vtk

    # Skip examples
    python tools/verify_build.py --skip-examples

    # Only build core library (C++ only, no Python)
    python tools/verify_build.py --only-core

    # Skip Python package verification
    python tools/verify_build.py --skip-python

    # Only test Python package (skip C++ cmake build)
    python tools/verify_build.py --only-python

    # Custom paths
    python tools/verify_build.py --work-dir /custom/path --install-prefix /custom/install

    # Keep build artifacts for inspection
    python tools/verify_build.py --keep
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import List, Optional, Tuple


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    """Return colored text if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


def print_step(name: str) -> None:
    """Print a step header."""
    print(f"\n{colored('==>', Colors.BLUE)} {colored(name, Colors.BOLD)}")


def print_pass(name: str) -> None:
    """Print a passing step."""
    print(f"  {colored('[PASS]', Colors.GREEN)} {name}")


def print_fail(name: str, error: str = "") -> None:
    """Print a failing step."""
    print(f"  {colored('[FAIL]', Colors.RED)} {name}")
    if error:
        for line in error.strip().split("\n"):
            print(f"         {line}")


def print_skip(name: str, reason: str = "") -> None:
    """Print a skipped step."""
    msg = f"  {colored('[SKIP]', Colors.YELLOW)} {name}"
    if reason:
        msg += f" ({reason})"
    print(msg)


# Test project CMakeLists.txt template (cmake install)
TEST_PROJECT_CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(test_trueform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(trueform REQUIRED CONFIG)

add_executable(test_app main.cpp)
target_link_libraries(test_app PRIVATE tf::trueform)
"""

# Test project CMakeLists.txt template (pip install via python -m trueform.cmake)
TEST_PROJECT_CMAKE_PIP = """\
cmake_minimum_required(VERSION 3.16)
project(test_trueform_pip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(trueform REQUIRED CONFIG)

add_executable(test_app main.cpp)
target_link_libraries(test_app PRIVATE tf::trueform)
"""

# Test project main.cpp template
TEST_PROJECT_MAIN = """\
#include <trueform/trueform.hpp>
#include <iostream>

int main() {
    auto pt = tf::make_point(1.0, 2.0, 3.0);
    std::cout << "trueform works! Point: "
              << pt[0] << ", " << pt[1] << ", " << pt[2] << std::endl;
    return 0;
}
"""

# Python test script template
TEST_PYTHON_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Test that trueform Python package works correctly.\"\"\"
import sys

def main():
    # Test import
    try:
        import trueform as tf
        print(f"trueform version: {tf.__version__}")
    except ImportError as e:
        print(f"Failed to import trueform: {e}")
        return 1

    # Test basic functionality - create a mesh and do a simple operation
    try:
        import numpy as np

        # Create a simple triangle mesh (a tetrahedron)
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float64)

        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ], dtype=np.int32)

        mesh = tf.Mesh(faces, points)
        print(f"Created mesh with {len(mesh.points)} points and {len(mesh.faces)} faces")

        # Test a spatial query
        point_cloud = tf.PointCloud(points)
        result = tf.neighbor_search(point_cloud, tf.Point([0.5, 0.5, 0.5]), 1.0)
        print(f"Neighbor search found {len(result)} points within radius 1.0")

        print("trueform Python package works!")
        return 0

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""


class BuildVerifier:
    """Verifies that trueform builds correctly from a clean git clone."""

    def __init__(
        self,
        work_dir: Path,
        install_prefix: Path,
        source_dir: Path,
        branch: Optional[str] = None,
    ):
        self.work_dir = work_dir
        self.install_prefix = install_prefix
        self.source_dir = source_dir
        self.branch = branch
        self.clone_dir = work_dir / "trueform"
        self.build_dir = self.clone_dir / "build"
        self.test_project_dir = work_dir / "test-project"
        self.venv_dir = work_dir / "venv"
        self.results: List[Tuple[str, bool, str]] = []  # (name, passed, error)

    def run_cmd(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        kwargs = {
            "cwd": cwd,
            "check": check,
        }
        if capture:
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.STDOUT
            kwargs["text"] = True
        return subprocess.run(cmd, **kwargs)

    def record_result(self, name: str, passed: bool, error: str = "") -> bool:
        """Record a test result."""
        self.results.append((name, passed, error))
        if passed:
            print_pass(name)
        else:
            print_fail(name, error)
        return passed

    def step_setup(self) -> bool:
        """Set up the work directory."""
        print_step("Setup")

        # Clean work directory if it exists
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
            except Exception as e:
                return self.record_result(
                    "Clean work directory", False, str(e)
                )
        self.record_result("Clean work directory", True)

        # Create work directory
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return self.record_result("Create work directory", False, str(e))
        self.record_result("Create work directory", True)

        return True

    def step_clone(self) -> bool:
        """Clone repository to work directory."""
        print_step("Clone Repository")

        clone_cmd = ["git", "clone", str(self.source_dir), str(self.clone_dir)]
        if self.branch:
            clone_cmd.extend(["--branch", self.branch])

        try:
            result = self.run_cmd(clone_cmd, cwd=self.work_dir)
            branch_msg = f" (branch: {self.branch})" if self.branch else ""
            return self.record_result(f"Clone repository{branch_msg}", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Clone repository", False, e.stdout if e.stdout else str(e)
            )

    def _get_parallel_jobs(self) -> int:
        """Get number of parallel build jobs."""
        return max(1, multiprocessing.cpu_count())

    def _cmake_build(self, target: str) -> subprocess.CompletedProcess:
        """Run cmake --build with parallel jobs."""
        return self.run_cmd(
            [
                "cmake", "--build", str(self.build_dir),
                "--target", target,
                "--parallel", str(self._get_parallel_jobs()),
            ],
            cwd=self.build_dir,
        )

    def step_build_core(self) -> bool:
        """Build core library."""
        print_step("Build Core Library")

        # Create build directory
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Configure
        try:
            result = self.run_cmd(
                [
                    "cmake",
                    "-S", str(self.clone_dir),
                    "-B", str(self.build_dir),
                    f"-DCMAKE_INSTALL_PREFIX={self.install_prefix}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DTF_BUILD_EXAMPLES=ON",
                    "-DTF_BUILD_VTK_INTEGRATION=ON",
                    "-DTF_BUILD_VTK_EXAMPLES=ON",
                ],
                cwd=self.build_dir,
            )
            self.record_result("Configure CMake", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Configure CMake", False, e.stdout if e.stdout else str(e)
            )

        # trueform is a header-only INTERFACE library, no build step needed
        # Configuration success means the library is ready
        return self.record_result("trueform (header-only)", True)

    def step_build_examples(self) -> bool:
        """Build C++ examples."""
        print_step("Build Examples")

        try:
            result = self._cmake_build("trueform_examples")
            return self.record_result("Build trueform_examples", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build trueform_examples", False,
                e.stdout if e.stdout else str(e)
            )

    def step_build_vtk(self) -> bool:
        """Build VTK integration library."""
        print_step("Build VTK Integration")

        try:
            result = self._cmake_build("trueform_vtk")
            return self.record_result("Build trueform_vtk", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build trueform_vtk", False,
                e.stdout if e.stdout else str(e)
            )

    def step_build_vtk_examples(self) -> bool:
        """Build VTK examples."""
        print_step("Build VTK Examples")

        try:
            result = self._cmake_build("trueform_vtk_examples")
            return self.record_result("Build trueform_vtk_examples", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build trueform_vtk_examples", False,
                e.stdout if e.stdout else str(e)
            )

    def step_install(self) -> bool:
        """Install to prefix."""
        print_step("Install")

        try:
            result = self.run_cmd(
                ["cmake", "--install", str(self.build_dir)],
                cwd=self.build_dir,
            )
            return self.record_result("Install trueform", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Install trueform", False, e.stdout if e.stdout else str(e)
            )

    def step_verify_install(self) -> bool:
        """Verify installed files exist."""
        print_step("Verify Installation")

        # Check for key installed files
        checks = [
            ("include/trueform/trueform.hpp", "Header files"),
            ("lib/cmake/trueform/trueformConfig.cmake", "CMake config"),
            ("lib/cmake/trueform/trueformTargets.cmake", "CMake targets"),
        ]

        all_passed = True
        for path, name in checks:
            full_path = self.install_prefix / path
            if full_path.exists():
                self.record_result(f"Verify {name}", True)
            else:
                self.record_result(
                    f"Verify {name}", False, f"Missing: {full_path}"
                )
                all_passed = False

        return all_passed

    def step_test_find_package(self) -> bool:
        """Test find_package from external project."""
        print_step("Test find_package")

        # Create test project directory
        self.test_project_dir.mkdir(parents=True, exist_ok=True)

        # Write CMakeLists.txt
        cmake_file = self.test_project_dir / "CMakeLists.txt"
        cmake_file.write_text(TEST_PROJECT_CMAKE)

        # Write main.cpp
        main_file = self.test_project_dir / "main.cpp"
        main_file.write_text(TEST_PROJECT_MAIN)

        test_build_dir = self.test_project_dir / "build"
        test_build_dir.mkdir(parents=True, exist_ok=True)

        # Configure test project
        try:
            result = self.run_cmd(
                [
                    "cmake",
                    "-S", str(self.test_project_dir),
                    "-B", str(test_build_dir),
                    f"-DCMAKE_PREFIX_PATH={self.install_prefix}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                cwd=test_build_dir,
            )
            self.record_result("Configure test project", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Configure test project", False,
                e.stdout if e.stdout else str(e)
            )

        # Build test project
        try:
            result = self.run_cmd(
                ["cmake", "--build", str(test_build_dir)],
                cwd=test_build_dir,
            )
            self.record_result("Build test project", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build test project", False, e.stdout if e.stdout else str(e)
            )

        # Run test executable
        test_exe = test_build_dir / "test_app"
        if not test_exe.exists():
            # Try Windows location
            test_exe = test_build_dir / "Release" / "test_app.exe"
            if not test_exe.exists():
                test_exe = test_build_dir / "Debug" / "test_app.exe"

        if not test_exe.exists():
            # On some systems, the executable might just be test_app without extension
            for candidate in test_build_dir.rglob("test_app*"):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    test_exe = candidate
                    break

        if not test_exe.exists():
            return self.record_result(
                "Run test executable", False, "Executable not found"
            )

        try:
            result = self.run_cmd([str(test_exe)], cwd=test_build_dir)
            if "trueform works!" in result.stdout:
                return self.record_result("Run test executable", True)
            else:
                return self.record_result(
                    "Run test executable", False,
                    f"Unexpected output: {result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Run test executable", False, e.stdout if e.stdout else str(e)
            )

    def step_test_find_package_vtk(self) -> bool:
        """Test find_package(trueform_vtk) from external project."""
        print_step("Test find_package (trueform_vtk)")

        vtk_test_dir = self.work_dir / "test-project-vtk"
        vtk_test_dir.mkdir(parents=True, exist_ok=True)

        # Write CMakeLists.txt for VTK test
        cmake_content = """\
cmake_minimum_required(VERSION 3.16)
project(test_trueform_vtk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(trueform_vtk REQUIRED CONFIG)

add_executable(test_app main.cpp)
target_link_libraries(test_app PRIVATE tf::vtk)
"""
        (vtk_test_dir / "CMakeLists.txt").write_text(cmake_content)

        # Write main.cpp for VTK test
        main_content = """\
#include <trueform/vtk.hpp>
#include <iostream>

int main() {
    std::cout << "trueform_vtk works!" << std::endl;
    return 0;
}
"""
        (vtk_test_dir / "main.cpp").write_text(main_content)

        vtk_build_dir = vtk_test_dir / "build"
        vtk_build_dir.mkdir(parents=True, exist_ok=True)

        # Configure
        try:
            result = self.run_cmd(
                [
                    "cmake",
                    "-S", str(vtk_test_dir),
                    "-B", str(vtk_build_dir),
                    f"-DCMAKE_PREFIX_PATH={self.install_prefix}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                cwd=vtk_build_dir,
            )
            self.record_result("Configure test project (vtk)", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Configure test project (vtk)", False,
                e.stdout if e.stdout else str(e)
            )

        # Build
        try:
            result = self.run_cmd(
                ["cmake", "--build", str(vtk_build_dir), "--parallel", str(self._get_parallel_jobs())],
                cwd=vtk_build_dir,
            )
            self.record_result("Build test project (vtk)", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build test project (vtk)", False,
                e.stdout if e.stdout else str(e)
            )

        # Find and run executable
        test_exe = vtk_build_dir / "test_app"
        if not test_exe.exists():
            for candidate in vtk_build_dir.rglob("test_app*"):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    test_exe = candidate
                    break

        if not test_exe.exists():
            return self.record_result(
                "Run test executable (vtk)", False, "Executable not found"
            )

        try:
            result = self.run_cmd([str(test_exe)], cwd=vtk_build_dir)
            if "trueform_vtk works!" in result.stdout:
                return self.record_result("Run test executable (vtk)", True)
            else:
                return self.record_result(
                    "Run test executable (vtk)", False,
                    f"Unexpected output: {result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Run test executable (vtk)", False,
                e.stdout if e.stdout else str(e)
            )

    def step_test_find_package_pip(self) -> bool:
        """Test find_package(trueform) using pip-installed package via python -m trueform.cmake."""
        print_step("Test find_package (via pip)")

        pip_test_dir = self.work_dir / "test-project-pip"
        pip_test_dir.mkdir(parents=True, exist_ok=True)

        # Write CMakeLists.txt
        (pip_test_dir / "CMakeLists.txt").write_text(TEST_PROJECT_CMAKE_PIP)
        (pip_test_dir / "main.cpp").write_text(TEST_PROJECT_MAIN)

        pip_build_dir = pip_test_dir / "build"
        pip_build_dir.mkdir(parents=True, exist_ok=True)

        # Get cmake dir from pip-installed package
        try:
            result = self.run_cmd(
                [str(self._get_venv_python()), "-m", "trueform.cmake"],
                cwd=self.work_dir,
            )
            cmake_dir = result.stdout.strip()
            self.record_result("Get trueform cmake dir", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Get trueform cmake dir", False,
                e.stdout if e.stdout else str(e)
            )

        # Configure with trueform_ROOT pointing to pip package
        try:
            result = self.run_cmd(
                [
                    "cmake",
                    "-S", str(pip_test_dir),
                    "-B", str(pip_build_dir),
                    f"-Dtrueform_ROOT={cmake_dir}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                cwd=pip_build_dir,
            )
            self.record_result("Configure test project (pip)", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Configure test project (pip)", False,
                e.stdout if e.stdout else str(e)
            )

        # Build
        try:
            result = self.run_cmd(
                ["cmake", "--build", str(pip_build_dir), "--parallel", str(self._get_parallel_jobs())],
                cwd=pip_build_dir,
            )
            self.record_result("Build test project (pip)", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build test project (pip)", False,
                e.stdout if e.stdout else str(e)
            )

        # Find and run executable
        test_exe = pip_build_dir / "test_app"
        if not test_exe.exists():
            for candidate in pip_build_dir.rglob("test_app*"):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    test_exe = candidate
                    break

        if not test_exe.exists():
            return self.record_result(
                "Run test executable (pip)", False, "Executable not found"
            )

        try:
            result = self.run_cmd([str(test_exe)], cwd=pip_build_dir)
            if "trueform works!" in result.stdout:
                return self.record_result("Run test executable (pip)", True)
            else:
                return self.record_result(
                    "Run test executable (pip)", False,
                    f"Unexpected output: {result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Run test executable (pip)", False,
                e.stdout if e.stdout else str(e)
            )

    def _get_venv_python(self) -> Path:
        """Get the path to the Python executable in the virtual environment."""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    def _get_venv_pip(self) -> Path:
        """Get the path to pip in the virtual environment."""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "pip.exe"
        return self.venv_dir / "bin" / "pip"

    def step_create_venv(self) -> bool:
        """Create a virtual environment for Python testing."""
        print_step("Create Python Virtual Environment")

        try:
            venv.create(self.venv_dir, with_pip=True)
            self.record_result("Create virtual environment", True)
        except Exception as e:
            return self.record_result(
                "Create virtual environment", False, str(e)
            )

        # Upgrade pip
        try:
            result = self.run_cmd(
                [str(self._get_venv_python()), "-m", "pip", "install", "--upgrade", "pip"],
                cwd=self.work_dir,
            )
            return self.record_result("Upgrade pip", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Upgrade pip", False, e.stdout if e.stdout else str(e)
            )

    def step_pip_install(self) -> bool:
        """Install trueform via pip."""
        print_step("Install trueform via pip")

        # Set parallel build jobs
        num_jobs = max(1, multiprocessing.cpu_count())

        env = os.environ.copy()
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(num_jobs)

        try:
            result = subprocess.run(
                [
                    str(self._get_venv_pip()),
                    "install",
                    "-v",
                    str(self.clone_dir),
                ],
                cwd=self.work_dir,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.record_result(f"pip install trueform (jobs={num_jobs})", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "pip install trueform", False,
                e.stdout if e.stdout else str(e)
            )

        # Install pytest for running tests
        try:
            result = subprocess.run(
                [str(self._get_venv_pip()), "install", "pytest"],
                cwd=self.work_dir,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return self.record_result("pip install pytest", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "pip install pytest", False,
                e.stdout if e.stdout else str(e)
            )

    def step_test_python_import(self) -> bool:
        """Test that trueform can be imported in Python."""
        print_step("Test Python Package")

        # Write test script
        test_script = self.work_dir / "test_trueform.py"
        test_script.write_text(TEST_PYTHON_SCRIPT)

        try:
            result = self.run_cmd(
                [str(self._get_venv_python()), str(test_script)],
                cwd=self.work_dir,
            )
            if "trueform Python package works!" in result.stdout:
                return self.record_result("Test Python package", True)
            else:
                return self.record_result(
                    "Test Python package", False,
                    f"Unexpected output:\n{result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Test Python package", False,
                e.stdout if e.stdout else str(e)
            )

    def step_run_python_tests(self) -> bool:
        """Run the Python test suite."""
        print_step("Run Python Tests")

        test_runner = self.clone_dir / "python" / "tests" / "run_tests.py"
        if not test_runner.exists():
            return self.record_result(
                "Run Python tests", False, "Test runner not found"
            )

        try:
            result = self.run_cmd(
                [str(self._get_venv_python()), str(test_runner)],
                cwd=self.clone_dir,
            )
            # Check for success message in output
            if "ALL TESTS PASSED" in result.stdout:
                # Extract summary line
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if "Passed:" in line:
                        return self.record_result(f"Python tests ({line.strip()})", True)
                return self.record_result("Run Python tests", True)
            else:
                # Extract failure count
                for line in result.stdout.strip().split("\n"):
                    if "Failed:" in line:
                        return self.record_result(
                            f"Run Python tests ({line.strip()})", False,
                            "Some tests failed"
                        )
                return self.record_result(
                    "Run Python tests", False, "Tests did not pass"
                )
        except subprocess.CalledProcessError as e:
            # Try to extract failure info from output
            output = e.stdout if e.stdout else str(e)
            failed_info = ""
            for line in output.split("\n"):
                if "Failed:" in line or "FAILED" in line:
                    failed_info = line.strip()
                    break
            return self.record_result(
                "Run Python tests", False,
                failed_info if failed_info else "Tests failed"
            )

    def run_all(
        self,
        skip_vtk: bool = False,
        skip_examples: bool = False,
        skip_python: bool = False,
        only_core: bool = False,
        only_python: bool = False,
    ) -> bool:
        """Run all verification steps."""
        # Setup and clone
        if not self.step_setup():
            return False
        if not self.step_clone():
            return False

        # C++ verification (skip if only_python)
        if not only_python:
            # Build core
            if not self.step_build_core():
                return False

            # Build examples
            if only_core:
                print_skip("Build Examples", "only-core mode")
            elif skip_examples:
                print_skip("Build Examples", "skipped by user")
            else:
                self.step_build_examples()  # Don't fail on examples

            # Build VTK
            vtk_ok = False
            if only_core or skip_vtk:
                reason = "only-core mode" if only_core else "skipped by user"
                print_skip("Build VTK Integration", reason)
                print_skip("Build VTK Examples", reason)
            else:
                vtk_ok = self.step_build_vtk()
                if vtk_ok:
                    self.step_build_vtk_examples()
                else:
                    print_skip("Build VTK Examples", "VTK build failed")

            # Install and verify
            if not self.step_install():
                return False
            if not self.step_verify_install():
                return False

            # Test find_package (trueform)
            if not self.step_test_find_package():
                return False

            # Test find_package (trueform_vtk) - only if VTK was built
            if not only_core and not skip_vtk and vtk_ok:
                self.step_test_find_package_vtk()  # Don't fail overall
            else:
                print_skip("Test find_package (vtk)", "VTK not built")
        else:
            print_skip("C++ Build", "only-python mode")
            print_skip("C++ Install", "only-python mode")
            print_skip("Test find_package", "only-python mode")
            print_skip("Test find_package (vtk)", "only-python mode")

        # Python verification (skip if only_core or skip_python)
        if only_core or skip_python:
            reason = "only-core mode" if only_core else "skipped by user"
            print_skip("Python Package", reason)
        else:
            if not self.step_create_venv():
                return False
            if not self.step_pip_install():
                return False
            if not self.step_test_python_import():
                return False
            # Test C++ linking via pip-installed package
            self.step_test_find_package_pip()  # Don't fail overall
            # Run Python tests (don't fail overall if tests fail, just record)
            self.step_run_python_tests()

        return True

    def print_summary(self) -> bool:
        """Print final summary and return overall success."""
        print(f"\n{colored('=' * 60, Colors.BOLD)}")
        print(colored("SUMMARY", Colors.BOLD))
        print(colored("=" * 60, Colors.BOLD))

        passed = sum(1 for _, p, _ in self.results if p)
        failed = sum(1 for _, p, _ in self.results if not p)
        total = len(self.results)

        for name, success, error in self.results:
            if success:
                print(f"  {colored('[PASS]', Colors.GREEN)} {name}")
            else:
                print(f"  {colored('[FAIL]', Colors.RED)} {name}")

        print()
        if failed == 0:
            print(
                colored(f"All {total} checks passed!", Colors.GREEN + Colors.BOLD)
            )
            return True
        else:
            print(
                colored(
                    f"{failed}/{total} checks failed",
                    Colors.RED + Colors.BOLD
                )
            )
            return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify trueform builds correctly from a clean git clone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/trueform-verify"),
        help="Working directory for clone and build (default: /tmp/trueform-verify)",
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
        help="Skip VTK integration build",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Skip examples build",
    )
    parser.add_argument(
        "--only-core",
        action="store_true",
        help="Only build core library (skip examples, VTK, and Python)",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python package verification",
    )
    parser.add_argument(
        "--only-python",
        action="store_true",
        help="Only test Python package (skip C++ cmake build)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to clone (default: current branch)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep build artifacts after verification",
    )

    args = parser.parse_args()

    # Determine source directory (parent of tools directory)
    script_dir = Path(__file__).resolve().parent
    source_dir = script_dir.parent

    # Set default install prefix if not specified
    install_prefix = args.install_prefix or (args.work_dir / "install")

    print(colored("Trueform Build Verification", Colors.BOLD + Colors.BLUE))
    print(f"  Source:  {source_dir}")
    print(f"  Work:    {args.work_dir}")
    print(f"  Install: {install_prefix}")
    if args.branch:
        print(f"  Branch:  {args.branch}")
    print()

    verifier = BuildVerifier(
        work_dir=args.work_dir,
        install_prefix=install_prefix,
        source_dir=source_dir,
        branch=args.branch,
    )

    try:
        verifier.run_all(
            skip_vtk=args.skip_vtk,
            skip_examples=args.skip_examples,
            skip_python=args.skip_python,
            only_core=args.only_core,
            only_python=args.only_python,
        )
    except KeyboardInterrupt:
        print(colored("\nInterrupted by user", Colors.YELLOW))
        return 130

    success = verifier.print_summary()

    # Clean up if requested
    if not args.keep and args.work_dir.exists():
        print(f"\nCleaning up {args.work_dir}...")
        try:
            shutil.rmtree(args.work_dir)
        except Exception as e:
            print(colored(f"Warning: Could not clean up: {e}", Colors.YELLOW))

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
