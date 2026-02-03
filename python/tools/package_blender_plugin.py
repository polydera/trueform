"""Package the Trueform Blender add-on zip from a trueform install or wheel."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _get_default_bpy_source() -> Path:
    """Get default bpy source path relative to this script (python/tools/ -> python/bpy/)."""
    return Path(__file__).resolve().parent.parent / "bpy"


def _get_default_plugin_dir() -> Path:
    """Get default plugin directory relative to this script."""
    return Path(__file__).resolve().parent.parent / "examples" / "bpy-plugin"


def _extract_trueform_from_wheel(wheel_path: Path, extract_dir: Path) -> Path:
    """Extract wheel and return path to trueform package inside."""
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(extract_dir)
    trueform_dir = extract_dir / "trueform"
    if not trueform_dir.is_dir():
        raise FileNotFoundError(f"trueform package not found in wheel: {wheel_path}")
    return trueform_dir


def _copy_plugin_files(plugin_dir: Path, bundle_root: Path) -> None:
    """Copy all Python files from plugin directory to bundle root."""
    for py_file in plugin_dir.rglob("*.py"):
        rel_path = py_file.relative_to(plugin_dir)
        dest = bundle_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, dest)


def package_blender_plugin(
    plugin_dir: Path,
    output: Path,
    wheel: Path,
    bpy_source: Path | None,
    bundle_name: str,
) -> Path:
    plugin_dir = plugin_dir.resolve()
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"plugin directory not found: {plugin_dir}")

    init_file = plugin_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"plugin __init__.py not found: {init_file}")

    wheel = wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"wheel not found: {wheel}")

    resolved_bpy_source = (bpy_source or _get_default_bpy_source()).resolve()
    if not resolved_bpy_source.is_dir():
        raise FileNotFoundError(f"bpy source not found: {resolved_bpy_source}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        resolved_trueform_root = _extract_trueform_from_wheel(wheel, tmp_path / "wheel_extract")

        bundle_root = tmp_path / bundle_name
        libs_dir = bundle_root / "libs" / "trueform"
        libs_dir.parent.mkdir(parents=True, exist_ok=True)

        _copy_plugin_files(plugin_dir, bundle_root)
        _copy_tree(resolved_trueform_root, libs_dir)
        _copy_tree(resolved_bpy_source, libs_dir / "bpy")

        archive_base = output.with_suffix("")
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=tmp_path,
            base_dir=bundle_name,
        )

    zip_path = output if output.suffix == ".zip" else output.with_suffix(".zip")
    if not zip_path.exists() and archive_base.with_suffix(".zip").exists():
        shutil.move(str(archive_base.with_suffix(".zip")), zip_path)

    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Trueform Blender add-on zip.")
    parser.add_argument("--plugin-dir", help="Path to add-on directory (default: python/examples/bpy-plugin/)")
    parser.add_argument("--output", required=True, help="Output zip path")
    parser.add_argument("--wheel", required=True, help="Path to trueform wheel")
    parser.add_argument("--bpy-source", help="Path to bpy source directory (default: python/bpy/)")
    parser.add_argument("--bundle-name", default="trueform-bpy", help="Top-level bundle folder")
    args = parser.parse_args()

    plugin_dir = Path(args.plugin_dir) if args.plugin_dir else _get_default_plugin_dir()

    zip_path = package_blender_plugin(
        plugin_dir=plugin_dir,
        output=Path(args.output),
        wheel=Path(args.wheel),
        bpy_source=Path(args.bpy_source) if args.bpy_source else None,
        bundle_name=args.bundle_name,
    )

    print(f"[Trueform] Blender plugin packaged at {zip_path}")


if __name__ == "__main__":
    main()
