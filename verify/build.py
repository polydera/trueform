#!/usr/bin/env python3
"""
Build and verify trueform from a clean clone.

This script:
1. Clones the repository to a work directory
2. Builds core, examples, tests, VTK, and Python bindings
3. Installs and verifies the installation
4. Tests find_package from external projects

Note: This script only builds. Use all.py to build and run tests.

Usage:
    python verify/build.py [options]

Examples:
    python verify/build.py                       # Full build and verify
    python verify/build.py --skip-vtk            # Skip VTK integration
    python verify/build.py --skip-python         # Skip Python bindings
    python verify/build.py --skip-examples       # Skip building examples
    python verify/build.py --keep                # Keep build artifacts
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from venv_utils import VenvInfo, create_venv


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


# Test project templates
TEST_PROJECT_CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(test_trueform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(trueform REQUIRED CONFIG)

add_executable(test_app main.cpp)
target_link_libraries(test_app PRIVATE tf::trueform)
"""

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


class BuildVerifier:
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
        self.python_build_dir = self.clone_dir / "build-python"
        self.test_project_dir = work_dir / "test-project"
        self.venv_dir = work_dir / "venv"
        self.venv_info: Optional[VenvInfo] = None
        self.results: List[Tuple[str, bool, str]] = []

    def run_cmd(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
        capture: bool = True,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        kwargs = {"cwd": cwd, "check": check}
        if env:
            kwargs["env"] = env
        if capture:
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.STDOUT
            kwargs["text"] = True
        return subprocess.run(cmd, **kwargs)

    def record_result(self, name: str, passed: bool, error: str = "") -> bool:
        self.results.append((name, passed, error))
        if passed:
            print_pass(name)
        else:
            print_fail(name, error)
        return passed

    def _get_parallel_jobs(self) -> int:
        return max(1, multiprocessing.cpu_count())

    def _cmake_build(self, target: str) -> subprocess.CompletedProcess:
        return self.run_cmd(
            [
                "cmake", "--build", str(self.build_dir),
                "--target", target,
                "--parallel", str(self._get_parallel_jobs()),
            ],
            cwd=self.build_dir,
        )


    # =========================================================================
    # Setup & Clone
    # =========================================================================
    def do_setup(self) -> bool:
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
            except Exception as e:
                return self.record_result("Clean work directory", False, str(e))
        self.record_result("Clean work directory", True)

        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return self.record_result("Create work directory", False, str(e))
        self.record_result("Create work directory", True)
        return True

    def do_clone(self) -> bool:
        clone_cmd = ["git", "clone", str(self.source_dir), str(self.clone_dir)]
        if self.branch:
            clone_cmd.extend(["--branch", self.branch])

        try:
            self.run_cmd(clone_cmd, cwd=self.work_dir)
            branch_msg = f" (branch: {self.branch})" if self.branch else ""
            return self.record_result(f"Clone repository{branch_msg}", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Clone repository", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Configure
    # =========================================================================
    def do_configure(self, skip_vtk: bool = False) -> bool:
        self.build_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            "-S", str(self.clone_dir),
            "-B", str(self.build_dir),
            f"-DCMAKE_INSTALL_PREFIX={self.install_prefix}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTF_BUILD_EXAMPLES=ON",
            "-DTF_BUILD_TESTS=ON",
        ]
        if not skip_vtk:
            cmake_args.extend([
                "-DTF_BUILD_VTK_INTEGRATION=ON",
                "-DTF_BUILD_VTK_EXAMPLES=ON",
            ])

        try:
            self.run_cmd(cmake_args, cwd=self.build_dir)
            return self.record_result("C++", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "C++", False, e.stdout if e.stdout else str(e)
            )

    def do_configure_python(self) -> bool:
        self.python_build_dir.mkdir(parents=True, exist_ok=True)

        # Use venv Python (venv must be created before calling this)
        cmake_args = [
            "cmake",
            "-S", str(self.clone_dir),
            "-B", str(self.python_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTF_BUILD_PYTHON=ON",
            "-DTF_USE_STATIC_TBB=ON",
            "-DTF_USE_SYSTEM_LIBS=OFF",
            f"-DPython3_EXECUTABLE={self.venv_info.python_exe}",
        ]

        try:
            self.run_cmd(cmake_args, cwd=self.python_build_dir, env=self.venv_info.get_env())
            return self.record_result("Python", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Python", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Build
    # =========================================================================
    def build_examples(self) -> bool:
        try:
            self._cmake_build("trueform_examples")
            return self.record_result("trueform_examples", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "trueform_examples", False, e.stdout if e.stdout else str(e)
            )

    def build_tests(self) -> bool:
        try:
            self._cmake_build("trueform_tests")
            return self.record_result("trueform_tests", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "trueform_tests", False, e.stdout if e.stdout else str(e)
            )

    def build_vtk(self) -> bool:
        try:
            self._cmake_build("trueform_vtk")
            return self.record_result("trueform_vtk", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "trueform_vtk", False, e.stdout if e.stdout else str(e)
            )

    def build_vtk_examples(self) -> bool:
        try:
            self._cmake_build("trueform_vtk_examples")
            return self.record_result("trueform_vtk_examples", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "trueform_vtk_examples", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Install
    # =========================================================================
    def install_cmake(self) -> bool:
        try:
            self.run_cmd(
                ["cmake", "--install", str(self.build_dir)],
                cwd=self.build_dir,
            )
            return self.record_result("cmake --install", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "cmake --install", False, e.stdout if e.stdout else str(e)
            )

    def install_pip(self) -> bool:
        """Install Python package from CMake build directory."""
        python_pkg_dir = self.python_build_dir / "python"
        if not python_pkg_dir.exists():
            return self.record_result("pip install", False, "Python build directory not found")

        # Create pyproject.toml for pip install of pre-built package
        pyproject = python_pkg_dir / "pyproject.toml"
        pyproject.write_text("""\
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "trueform"
version = "0.0.0"
description = "Real-time geometric processing on NumPy arrays. Easy to use, robust on real-world data."
authors = [
  { name = "Ziga Sajovic", email = "info@polydera.com" },
  { name = "XLAB" },
]
keywords = [
    "mesh",
    "geometry",
    "computational-geometry",
    "mesh-processing",
    "mesh-boolean",
    "collision-detection",
    "point-cloud",
    "csg",
    "numpy",
]
dependencies = ["numpy>=1.23"]

[tool.setuptools.packages.find]
where = ["."]
include = ["trueform*"]

[tool.setuptools.package-data]
trueform = [
    "*.so",
    "*.pyd",
    "*.dylib",
    "*.dll",
    "include/**/*.hpp",
    "include/**/*.h",
    "cmake_config/*.cmake",
]
""")

        try:
            self.venv_info.run_pip(["install", str(python_pkg_dir)], cwd=self.work_dir)
            return self.record_result("pip install (from build)", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "pip install", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Verify trueform
    # =========================================================================
    def verify_trueform(self) -> bool:
        checks = [
            ("include/trueform/trueform.hpp", "Header files"),
            ("lib/cmake/trueform/trueformConfig.cmake", "CMake config"),
            ("lib/cmake/trueform/trueformTargets.cmake", "CMake targets"),
        ]

        all_passed = True
        for path, name in checks:
            full_path = self.install_prefix / path
            if full_path.exists():
                self.record_result(name, True)
            else:
                self.record_result(name, False, f"Missing: {full_path}")
                all_passed = False
        return all_passed

    # =========================================================================
    # Verify trueform_vtk
    # =========================================================================
    def verify_trueform_vtk(self) -> bool:
        checks = [
            ("include/trueform/vtk.hpp", "VTK header"),
            ("lib/cmake/trueform_vtk/trueform_vtkConfig.cmake", "VTK CMake config"),
        ]

        all_passed = True
        for path, name in checks:
            full_path = self.install_prefix / path
            if full_path.exists():
                self.record_result(name, True)
            else:
                self.record_result(name, False, f"Missing: {full_path}")
                all_passed = False
        return all_passed

    # =========================================================================
    # Verify pip
    # =========================================================================
    def verify_pip(self) -> bool:
        # Check import
        try:
            result = self.venv_info.run_python(
                ["-c", "import trueform as tf; print(tf.__version__)"],
                cwd=self.work_dir,
            )
            self.record_result(f"import trueform (v{result.stdout.strip()})", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "import trueform", False, e.stdout if e.stdout else str(e)
            )

        # Check python -m trueform.cmake
        try:
            self.venv_info.run_python(["-m", "trueform.cmake"], cwd=self.work_dir)
            self.record_result("trueform.cmake", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "trueform.cmake", False, e.stdout if e.stdout else str(e)
            )

        return True

    # =========================================================================
    # Test find_package
    # =========================================================================
    def test_find_package(self) -> bool:
        self.test_project_dir.mkdir(parents=True, exist_ok=True)
        (self.test_project_dir / "CMakeLists.txt").write_text(TEST_PROJECT_CMAKE)
        (self.test_project_dir / "main.cpp").write_text(TEST_PROJECT_MAIN)

        test_build_dir = self.test_project_dir / "build"
        test_build_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.run_cmd(
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
                "Configure test project", False, e.stdout if e.stdout else str(e)
            )

        try:
            self.run_cmd(
                ["cmake", "--build", str(test_build_dir)],
                cwd=test_build_dir,
            )
            self.record_result("Build test project", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build test project", False, e.stdout if e.stdout else str(e)
            )

        test_exe = test_build_dir / "test_app"
        if not test_exe.exists():
            for candidate in test_build_dir.rglob("test_app*"):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    test_exe = candidate
                    break

        if not test_exe.exists():
            return self.record_result("Run test executable", False, "Executable not found")

        try:
            result = self.run_cmd([str(test_exe)], cwd=test_build_dir)
            if "trueform works!" in result.stdout:
                return self.record_result("Run test executable", True)
            else:
                return self.record_result(
                    "Run test executable", False, f"Unexpected output: {result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Run test executable", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Test find_package (vtk)
    # =========================================================================
    def test_find_package_vtk(self) -> bool:
        vtk_test_dir = self.work_dir / "test-project-vtk"
        vtk_test_dir.mkdir(parents=True, exist_ok=True)

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

        try:
            self.run_cmd(
                [
                    "cmake",
                    "-S", str(vtk_test_dir),
                    "-B", str(vtk_build_dir),
                    f"-DCMAKE_PREFIX_PATH={self.install_prefix}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                cwd=vtk_build_dir,
            )
            self.record_result("Configure test project", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Configure test project", False, e.stdout if e.stdout else str(e)
            )

        try:
            self.run_cmd(
                ["cmake", "--build", str(vtk_build_dir), "--parallel", str(self._get_parallel_jobs())],
                cwd=vtk_build_dir,
            )
            self.record_result("Build test project", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build test project", False, e.stdout if e.stdout else str(e)
            )

        test_exe = vtk_build_dir / "test_app"
        if not test_exe.exists():
            for candidate in vtk_build_dir.rglob("test_app*"):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    test_exe = candidate
                    break

        if not test_exe.exists():
            return self.record_result("Run test executable", False, "Executable not found")

        try:
            result = self.run_cmd([str(test_exe)], cwd=vtk_build_dir)
            if "trueform_vtk works!" in result.stdout:
                return self.record_result("Run test executable", True)
            else:
                return self.record_result(
                    "Run test executable", False, f"Unexpected output: {result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Run test executable", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Test find_package (pip)
    # =========================================================================
    def test_find_package_pip(self) -> bool:
        pip_test_dir = self.work_dir / "test-project-pip"
        pip_test_dir.mkdir(parents=True, exist_ok=True)

        (pip_test_dir / "CMakeLists.txt").write_text(TEST_PROJECT_CMAKE)
        (pip_test_dir / "main.cpp").write_text(TEST_PROJECT_MAIN)

        pip_build_dir = pip_test_dir / "build"
        pip_build_dir.mkdir(parents=True, exist_ok=True)

        # Get cmake dir from pip-installed package
        try:
            result = self.venv_info.run_python(["-m", "trueform.cmake"], cwd=self.work_dir)
            cmake_dir = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Configure test project", False,
                f"Could not get cmake dir: {e.stdout if e.stdout else str(e)}"
            )

        try:
            self.run_cmd(
                [
                    "cmake",
                    "-S", str(pip_test_dir),
                    "-B", str(pip_build_dir),
                    f"-Dtrueform_ROOT={cmake_dir}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                cwd=pip_build_dir,
            )
            self.record_result("Configure test project", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Configure test project", False, e.stdout if e.stdout else str(e)
            )

        try:
            self.run_cmd(
                ["cmake", "--build", str(pip_build_dir), "--parallel", str(self._get_parallel_jobs())],
                cwd=pip_build_dir,
            )
            self.record_result("Build test project", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Build test project", False, e.stdout if e.stdout else str(e)
            )

        test_exe = pip_build_dir / "test_app"
        if not test_exe.exists():
            for candidate in pip_build_dir.rglob("test_app*"):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    test_exe = candidate
                    break

        if not test_exe.exists():
            return self.record_result("Run test executable", False, "Executable not found")

        try:
            result = self.run_cmd([str(test_exe)], cwd=pip_build_dir)
            if "trueform works!" in result.stdout:
                return self.record_result("Run test executable", True)
            else:
                return self.record_result(
                    "Run test executable", False, f"Unexpected output: {result.stdout}"
                )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "Run test executable", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Python venv & wheel
    # =========================================================================
    def do_create_venv(self) -> bool:
        """Create venv and store info in self.venv_info."""
        try:
            self.venv_info = create_venv(self.venv_dir, with_pip=True, upgrade_pip=True)
            return self.record_result(f"Create venv ({self.venv_info.python_exe.name})", True)
        except Exception as e:
            return self.record_result("Create venv", False, str(e))

    def build_python(self) -> bool:
        """Build and install Python bindings."""
        try:
            self.venv_info.run(
                [
                    "cmake", "--build", str(self.python_build_dir),
                    "--target", "trueform_python",
                    "--parallel", str(self._get_parallel_jobs()),
                ],
                cwd=self.python_build_dir,
            )
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "trueform_python", False, e.stdout if e.stdout else str(e)
            )

        # Install to python build dir (puts headers and cmake config in place)
        python_pkg_dir = self.python_build_dir / "python"
        try:
            self.venv_info.run(
                [
                    "cmake", "--install", str(self.python_build_dir),
                    "--prefix", str(python_pkg_dir),
                ],
                cwd=self.python_build_dir,
            )
            return self.record_result("trueform_python", True)
        except subprocess.CalledProcessError as e:
            return self.record_result(
                "trueform_python", False, e.stdout if e.stdout else str(e)
            )

    # =========================================================================
    # Main orchestration
    # =========================================================================
    def run_all(
        self,
        skip_vtk: bool = False,
        skip_python: bool = False,
        skip_examples: bool = False,
    ) -> bool:
        print_step("Setup")
        if not self.do_setup():
            return False

        print_step("Clone")
        if not self.do_clone():
            return False

        print_step("Configure")
        if not self.do_configure(skip_vtk=skip_vtk):
            return False

        # Create venv before configuring Python so we use venv's Python
        venv_ok = False
        if not skip_python:
            venv_ok = self.do_create_venv()
            if venv_ok:
                if not self.do_configure_python():
                    return False
            else:
                print_skip("Python configure", "venv creation failed")
        else:
            print_skip("Python", "skipped by user")

        print_step("Build")
        if skip_examples:
            print_skip("trueform_examples", "skipped by user")
        else:
            self.build_examples()
        self.build_tests()

        vtk_ok = False
        if skip_vtk:
            print_skip("trueform_vtk", "skipped by user")
            print_skip("trueform_vtk_examples", "skipped by user")
        else:
            vtk_ok = self.build_vtk()
            if vtk_ok:
                if skip_examples:
                    print_skip("trueform_vtk_examples", "skipped by user")
                else:
                    self.build_vtk_examples()
            else:
                print_skip("trueform_vtk_examples", "VTK build failed")

        python_ok = False
        if skip_python:
            print_skip("trueform_python", "skipped by user")
        elif venv_ok:
            python_ok = self.build_python()
        else:
            print_skip("trueform_python", "venv creation failed")

        print_step("Install")
        if not self.install_cmake():
            return False

        pip_ok = False
        if skip_python:
            print_skip("pip install", "skipped by user")
        elif python_ok and venv_ok:
            pip_ok = self.install_pip()
        else:
            print_skip("pip install", "Python build failed")

        print_step("Verify trueform")
        self.verify_trueform()

        if skip_vtk:
            print_step("Verify trueform_vtk")
            print_skip("VTK verification", "skipped by user")
        elif vtk_ok:
            print_step("Verify trueform_vtk")
            self.verify_trueform_vtk()

        if skip_python:
            print_step("Verify pip")
            print_skip("pip verification", "skipped by user")
        elif pip_ok and venv_ok:
            print_step("Verify pip")
            self.verify_pip()

        print_step("Test find_package")
        self.test_find_package()

        if skip_vtk:
            print_step("Test find_package (vtk)")
            print_skip("VTK find_package test", "skipped by user")
        elif vtk_ok:
            print_step("Test find_package (vtk)")
            self.test_find_package_vtk()

        if skip_python:
            print_step("Test find_package (pip)")
            print_skip("pip find_package test", "skipped by user")
        elif pip_ok and venv_ok:
            print_step("Test find_package (pip)")
            self.test_find_package_pip()

        return True

    def print_summary(self) -> bool:
        print_step("Summary")

        failed = sum(1 for _, p, _ in self.results if not p)
        total = len(self.results)

        for name, success, error in self.results:
            if success:
                print(f"  {colored('[PASS]', Colors.GREEN)} {name}")
            else:
                print(f"  {colored('[FAIL]', Colors.RED)} {name}")

        print()
        if failed == 0:
            print(colored(f"All {total} checks passed!", Colors.GREEN + Colors.BOLD))
            return True
        else:
            print(colored(f"{failed}/{total} checks failed", Colors.RED + Colors.BOLD))
            return False


def run_build_cpp_only(
    work_dir: Path = None,
    skip_vtk: bool = False,
    skip_examples: bool = False,
    branch: str = None,
    keep: bool = False,
    source_dir: Path = None,
) -> bool:
    """
    Build and verify C++ only (no Python).

    Returns True if all checks passed, False otherwise.
    """
    if work_dir is None:
        work_dir = Path(tempfile.gettempdir()) / "trueform-verify"
    if source_dir is None:
        source_dir = Path(__file__).resolve().parent.parent
    install_prefix = work_dir / "install"

    print(colored("Trueform C++ Build", Colors.BOLD + Colors.BLUE))
    print(f"  Source:  {source_dir}")
    print(f"  Work:    {work_dir}")
    print(f"  Install: {install_prefix}")
    if branch:
        print(f"  Branch:  {branch}")
    print()

    verifier = BuildVerifier(
        work_dir=work_dir,
        install_prefix=install_prefix,
        source_dir=source_dir,
        branch=branch,
    )

    try:
        print_step("Setup")
        if not verifier.do_setup():
            return False

        print_step("Clone")
        if not verifier.do_clone():
            return False

        print_step("Configure")
        if not verifier.do_configure(skip_vtk=skip_vtk):
            return False

        print_step("Build")
        if skip_examples:
            print_skip("trueform_examples", "skipped by user")
        else:
            verifier.build_examples()
        verifier.build_tests()

        if skip_vtk:
            print_skip("trueform_vtk", "skipped by user")
            print_skip("trueform_vtk_examples", "skipped by user")
        else:
            vtk_ok = verifier.build_vtk()
            if vtk_ok:
                if skip_examples:
                    print_skip("trueform_vtk_examples", "skipped by user")
                else:
                    verifier.build_vtk_examples()
            else:
                print_skip("trueform_vtk_examples", "VTK build failed")

        print_step("Install")
        if not verifier.install_cmake():
            return False

        print_step("Verify trueform")
        verifier.verify_trueform()

        if skip_vtk:
            print_step("Verify trueform_vtk")
            print_skip("VTK verification", "skipped by user")
        else:
            print_step("Verify trueform_vtk")
            verifier.verify_trueform_vtk()

        print_step("Test find_package")
        verifier.test_find_package()

        if skip_vtk:
            print_step("Test find_package (vtk)")
            print_skip("VTK find_package test", "skipped by user")
        else:
            print_step("Test find_package (vtk)")
            verifier.test_find_package_vtk()

    except KeyboardInterrupt:
        print(colored("\nInterrupted by user", Colors.YELLOW))
        return False

    print_step("Summary")
    failed = sum(1 for _, p, _ in verifier.results if not p)
    total = len(verifier.results)

    for name, success, _ in verifier.results:
        if success:
            print(f"  {colored('[PASS]', Colors.GREEN)} {name}")
        else:
            print(f"  {colored('[FAIL]', Colors.RED)} {name}")

    print()
    if failed == 0:
        print(colored(f"All {total} checks passed!", Colors.GREEN + Colors.BOLD))
        return True
    else:
        print(colored(f"{failed}/{total} checks failed", Colors.RED + Colors.BOLD))
        return False


def run_build_python_only(
    work_dir: Path = None,
    branch: str = None,
    keep: bool = False,
    source_dir: Path = None,
) -> bool:
    """
    Build Python bindings only (for CI).

    Returns True if all checks passed, False otherwise.
    """
    if work_dir is None:
        work_dir = Path(tempfile.gettempdir()) / "trueform-verify"
    if source_dir is None:
        source_dir = Path(__file__).resolve().parent.parent

    print(colored("Trueform Python Build", Colors.BOLD + Colors.BLUE))
    print(f"  Source: {source_dir}")
    print(f"  Work:   {work_dir}")
    if branch:
        print(f"  Branch: {branch}")
    print()

    verifier = BuildVerifier(
        work_dir=work_dir,
        install_prefix=work_dir / "install",
        source_dir=source_dir,
        branch=branch,
    )

    try:
        print_step("Setup")
        if not verifier.do_setup():
            return False

        print_step("Clone")
        if not verifier.do_clone():
            return False

        print_step("Create venv")
        if not verifier.do_create_venv():
            return False

        print_step("Configure")
        if not verifier.do_configure_python():
            return False

        print_step("Build")
        if not verifier.build_python():
            return False

        print_step("Install")
        if not verifier.install_pip():
            return False

        print_step("Verify")
        verifier.verify_pip()

    except KeyboardInterrupt:
        print(colored("\nInterrupted by user", Colors.YELLOW))
        return False

    print_step("Summary")
    failed = sum(1 for _, p, _ in verifier.results if not p)
    total = len(verifier.results)

    for name, success, _ in verifier.results:
        if success:
            print(f"  {colored('[PASS]', Colors.GREEN)} {name}")
        else:
            print(f"  {colored('[FAIL]', Colors.RED)} {name}")

    print()
    if failed == 0:
        print(colored(f"All {total} checks passed!", Colors.GREEN + Colors.BOLD))
        return True
    else:
        print(colored(f"{failed}/{total} checks failed", Colors.RED + Colors.BOLD))
        return False


def run_build(
    work_dir: Path = None,
    install_prefix: Path = None,
    skip_vtk: bool = False,
    skip_python: bool = False,
    skip_examples: bool = False,
    branch: str = None,
    keep: bool = False,
    source_dir: Path = None,
) -> bool:
    """
    Build and install trueform from a clean clone.

    Returns True if all checks passed, False otherwise.
    """
    if work_dir is None:
        work_dir = Path(tempfile.gettempdir()) / "trueform-verify"
    if source_dir is None:
        source_dir = Path(__file__).resolve().parent.parent
    if install_prefix is None:
        install_prefix = work_dir / "install"

    print(colored("Trueform Build Verification", Colors.BOLD + Colors.BLUE))
    print(f"  Source:  {source_dir}")
    print(f"  Work:    {work_dir}")
    print(f"  Install: {install_prefix}")
    if branch:
        print(f"  Branch:  {branch}")
    print()

    verifier = BuildVerifier(
        work_dir=work_dir,
        install_prefix=install_prefix,
        source_dir=source_dir,
        branch=branch,
    )

    try:
        verifier.run_all(
            skip_vtk=skip_vtk,
            skip_python=skip_python,
            skip_examples=skip_examples,
        )
    except KeyboardInterrupt:
        print(colored("\nInterrupted by user", Colors.YELLOW))
        return False

    success = verifier.print_summary()

    if not keep and work_dir.exists():
        print(f"\nCleaning up {work_dir}...")
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            print(colored(f"Warning: Could not clean up: {e}", Colors.YELLOW))

    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and install trueform from a clean clone.",
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

    success = run_build(
        work_dir=args.work_dir,
        install_prefix=args.install_prefix,
        skip_vtk=args.skip_vtk,
        skip_python=args.skip_python,
        skip_examples=args.skip_examples,
        branch=args.branch,
        keep=args.keep,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
