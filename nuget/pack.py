#!/usr/bin/env python3
"""
Pack the trueform NuGet package.

Reads the version from CMakeLists.txt, collects headers and license files,
and produces a .nupkg (zip) file. No external tools required.

Usage:
    python nuget/pack.py [--output-dir DIR]
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

NUGET_DIR = Path(__file__).resolve().parent
ROOT = NUGET_DIR.parent
INCLUDE_DIR = ROOT / "include"


def read_version() -> str:
    cmake = ROOT / "CMakeLists.txt"
    match = re.search(
        r"project\(trueform\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
        cmake.read_text(),
    )
    if not match:
        print("ERROR: Could not read version from CMakeLists.txt", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description="Pack trueform NuGet package")
    parser.add_argument(
        "--output-dir", "-o",
        default=str(ROOT / "dist-nuget"),
        help="Output directory for .nupkg (default: dist-nuget/)",
    )
    args = parser.parse_args()

    version = read_version()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkg_name = f"polydera.trueform.{version}.nupkg"
    pkg_path = output_dir / pkg_name

    # Fill version in nuspec
    nuspec_content = (
        NUGET_DIR / "trueform.nuspec").read_text().replace("$version$", version)

    print(f"Packing polydera.trueform {version}...")

    with zipfile.ZipFile(pkg_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # NuGet metadata
        zf.write(NUGET_DIR / "content_types.xml", "[Content_Types].xml")
        zf.write(NUGET_DIR / "rels.xml", "_rels/.rels")
        zf.writestr("polydera.trueform.nuspec", nuspec_content)

        # MSBuild targets
        zf.write(NUGET_DIR / "trueform.targets",
                 "build/native/polydera.trueform.targets")

        # Headers. version.hpp is generated: the working tree's copy may
        # be stale, so it is rendered here from the version this pack
        # reads, and whatever lies on disk is skipped.
        version_rel = "trueform/version.hpp"
        junk = (".bak", ".orig", ".tmp", ".swp")
        for header in sorted(INCLUDE_DIR.rglob("*")):
            if not header.is_file():
                continue
            if header.name.endswith(junk) or header.name == ".DS_Store":
                continue
            rel = str(header.relative_to(INCLUDE_DIR))
            if rel == version_rel:
                continue
            zf.write(header, "build/native/include/" + rel)

        major, minor, patch = version.split(".")
        version_hpp = (ROOT / "cmake" / "version.hpp.in").read_text()
        version_hpp = (
            version_hpp.replace("@PROJECT_VERSION_MAJOR@", major)
            .replace("@PROJECT_VERSION_MINOR@", minor)
            .replace("@PROJECT_VERSION_PATCH@", patch)
            .replace("@PROJECT_VERSION@", version)
        )
        zf.writestr("build/native/include/" + version_rel, version_hpp)

        # README
        zf.write(NUGET_DIR / "README.md", "README.md")

        # Release notes
        zf.write(ROOT / "RELEASE_NOTES.md", "RELEASE_NOTES.md")

        # Licenses
        zf.write(ROOT / "LICENSE", "LICENSE")
        zf.write(ROOT / "LICENSE.noncommercial", "LICENSE.noncommercial")

    print(f"Done. {pkg_path}")


if __name__ == "__main__":
    main()
