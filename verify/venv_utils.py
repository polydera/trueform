#!/usr/bin/env python3
"""
Venv utilities for trueform verification scripts.

Provides proper venv creation and environment handling without
platform-specific hardcoding.
"""

import multiprocessing
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional


class VenvInfo:
    """Information about a virtual environment."""

    def __init__(self, venv_dir: Path, bin_path: Path, python_exe: Path, bin_name: str):
        self.venv_dir = venv_dir
        self.bin_path = bin_path
        self.python_exe = python_exe
        self.bin_name = bin_name

    def get_env(self) -> dict:
        """Get environment dict with venv activated."""
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(self.venv_dir)
        env["PATH"] = str(self.bin_path) + os.pathsep + env.get("PATH", "")
        env.pop("PYTHONHOME", None)
        # Enable parallel cmake builds by default
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in env:
            env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(multiprocessing.cpu_count())
        return env

    def run(
        self,
        cmd: list,
        cwd: Path = None,
        capture: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command with venv environment."""
        kwargs = {"cwd": cwd, "check": check, "env": self.get_env()}
        if capture:
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.STDOUT
            kwargs["text"] = True
        return subprocess.run(cmd, **kwargs)

    def run_python(
        self,
        args: list,
        cwd: Path = None,
        capture: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run venv Python with given arguments."""
        return self.run([str(self.python_exe)] + args, cwd=cwd, capture=capture, check=check)

    def run_pip(
        self,
        args: list,
        cwd: Path = None,
        capture: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run venv pip with given arguments (via python -m pip for portability)."""
        return self.run_python(["-m", "pip"] + args, cwd=cwd, capture=capture, check=check)


class _VenvBuilder(venv.EnvBuilder):
    """Custom venv builder that captures the context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = None

    def post_setup(self, context):
        self.context = context


def create_venv(venv_dir: Path, with_pip: bool = True, upgrade_pip: bool = True) -> VenvInfo:
    """
    Create a virtual environment and return its info.

    Args:
        venv_dir: Directory to create venv in
        with_pip: Include pip in the venv
        upgrade_pip: Upgrade pip after creation

    Returns:
        VenvInfo object with paths and utilities
    """
    builder = _VenvBuilder(with_pip=with_pip)
    builder.create(str(venv_dir))

    ctx = builder.context
    info = VenvInfo(
        venv_dir=Path(ctx.env_dir),
        bin_path=Path(ctx.bin_path),
        python_exe=Path(ctx.env_exe),
        bin_name=ctx.bin_name,
    )

    if upgrade_pip and with_pip:
        info.run_pip(["install", "--upgrade", "pip"])

    return info


def get_venv_info(venv_dir: Path) -> Optional[VenvInfo]:
    """
    Get info for an existing venv.

    Args:
        venv_dir: Path to existing venv

    Returns:
        VenvInfo if venv exists and is valid, None otherwise
    """
    if not venv_dir.exists():
        return None

    # Determine bin path and python exe based on platform
    if sys.platform == "win32":
        bin_name = "Scripts"
        bin_path = venv_dir / "Scripts"
        python_exe = bin_path / "python.exe"
    else:
        bin_name = "bin"
        bin_path = venv_dir / "bin"
        python_exe = bin_path / "python"

    if not python_exe.exists():
        return None

    return VenvInfo(
        venv_dir=venv_dir,
        bin_path=bin_path,
        python_exe=python_exe,
        bin_name=bin_name,
    )
