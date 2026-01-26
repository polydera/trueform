"""Package the Trueform Blender add-on zip from a trueform install."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def _resolve_trueform_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"trueform root not found: {root}")
        return root

    import trueform

    return Path(trueform.__file__).resolve().parent


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _get_default_bpy_source() -> Path:
    """Get default bpy source path relative to this script (python/tools/ -> python/bpy/)."""
    return Path(__file__).resolve().parent.parent / "bpy"


def package_blender_plugin(
    plugin_init: Path,
    output: Path,
    trueform_root: Path | None,
    bpy_source: Path | None,
    bundle_name: str,
) -> Path:
    plugin_init = plugin_init.resolve()
    if not plugin_init.is_file():
        raise FileNotFoundError(f"plugin init not found: {plugin_init}")

    resolved_trueform_root = _resolve_trueform_root(str(trueform_root) if trueform_root else None)

    # Resolve bpy source (default: python/bpy/ relative to this script)
    resolved_bpy_source = (bpy_source or _get_default_bpy_source()).resolve()
    if not resolved_bpy_source.is_dir():
        raise FileNotFoundError(f"bpy source not found: {resolved_bpy_source}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        bundle_root = tmp_path / bundle_name
        libs_dir = bundle_root / "libs" / "trueform"
        libs_dir.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(plugin_init, bundle_root / "__init__.py")
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
    parser.add_argument("--plugin-init", required=True, help="Path to add-on __init__.py")
    parser.add_argument("--output", required=True, help="Output zip path")
    parser.add_argument("--trueform-root", help="Path to trueform package root (default: find via import)")
    parser.add_argument("--bpy-source", help="Path to bpy source directory (default: python/bpy/)")
    parser.add_argument("--bundle-name", default="trueform-bpy", help="Top-level bundle folder")
    args = parser.parse_args()

    zip_path = package_blender_plugin(
        plugin_init=Path(args.plugin_init),
        output=Path(args.output),
        trueform_root=Path(args.trueform_root) if args.trueform_root else None,
        bpy_source=Path(args.bpy_source) if args.bpy_source else None,
        bundle_name=args.bundle_name,
    )

    print(f"[Trueform] Blender plugin packaged at {zip_path}")


if __name__ == "__main__":
    main()
