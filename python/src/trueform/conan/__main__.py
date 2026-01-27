"""
Conan integration CLI for TrueForm

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import shutil
import subprocess
import sys

from . import get_conan_dir

USAGE = """\
Usage: python -m trueform.conan create [conan-options...]
       python -m trueform.conan --path

Commands:
  create [options]    Add trueform to Conan cache
                      All options are passed directly to 'conan create'
                      Default: --build=missing if no --build specified

  --path              Print path to conanfile directory

Examples:
  python -m trueform.conan create
  python -m trueform.conan create --build=*
  python -m trueform.conan create -pr:h myprofile
  conan create $(python -m trueform.conan --path) --build=missing
"""


def main() -> None:
    args = sys.argv[1:]

    # --path flag: print path and exit
    if "--path" in args:
        print(get_conan_dir())
        return

    # create command: pass remaining args to conan
    if args and args[0] == "create":
        conan_create(args[1:])
        return

    # No command or unknown - show usage
    print(USAGE)


def conan_create(extra_args: list) -> None:
    """Run conan create to add trueform to local Conan cache."""
    if shutil.which("conan") is None:
        print("Error: conan not found.", file=sys.stderr)
        print("Install with: pip install conan", file=sys.stderr)
        sys.exit(1)

    conan_dir = get_conan_dir()
    cmd = ["conan", "create", conan_dir]

    # Default to --build=missing if no --build flag provided
    if not any(arg.startswith("--build") for arg in extra_args):
        cmd.append("--build=missing")

    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
