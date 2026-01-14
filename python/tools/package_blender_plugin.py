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


def package_blender_plugin(
    plugin_init: Path,
    output: Path,
    trueform_root: Path | None,
    bundle_name: str,
) -> Path:
    plugin_init = plugin_init.resolve()
    if not plugin_init.is_file():
        raise FileNotFoundError(f"plugin init not found: {plugin_init}")

    resolved_trueform_root = _resolve_trueform_root(str(trueform_root) if trueform_root else None)

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        bundle_root = tmp_path / bundle_name
        libs_dir = bundle_root / "libs" / "trueform"
        libs_dir.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(plugin_init, bundle_root / "__init__.py")
        _copy_tree(resolved_trueform_root, libs_dir)

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
    parser.add_argument("--trueform-root", help="Path to trueform package root")
    parser.add_argument("--bundle-name", default="trueform-bpy", help="Top-level bundle folder")
    args = parser.parse_args()

    zip_path = package_blender_plugin(
        plugin_init=Path(args.plugin_init),
        output=Path(args.output),
        trueform_root=Path(args.trueform_root) if args.trueform_root else None,
        bundle_name=args.bundle_name,
    )

    print(f"[Trueform] Blender plugin packaged at {zip_path}")


if __name__ == "__main__":
    main()
