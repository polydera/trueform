#!/usr/bin/env python3
"""
Consolidate benchmark results by merging per-library CSV files into unified per-test files,
and optionally export to JSON for the documentation charts.

Example:
    results/cut/boolean-tf.csv
    results/cut/boolean-cgal.csv
    results/cut/boolean-igl.csv

    -> results/cut/consolidated-boolean.csv (with 'library' column: tf, cgal, igl)
    -> docs/benchmarks/boolean.json (pivoted wide format for charts)

Consolidated files are prefixed with 'consolidated-' and are skipped in subsequent runs.

Usage:
    python consolidate_results.py
    python consolidate_results.py --remove-originals
    python consolidate_results.py --json ../docs/benchmarks
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse benchmark filename to extract test name and library name.

    The library name is everything after the LAST dash before .csv

    Args:
        filename: CSV filename (e.g., "point_cloud-knn-tf.csv")

    Returns:
        Tuple of (test_name, library_name)

    Examples:
        "boolean-tf.csv" -> ("boolean", "tf")
        "point_cloud-knn-tf.csv" -> ("point_cloud-knn", "tf")
        "polygons_to_polygons-closest_point-fcl.csv" -> ("polygons_to_polygons-closest_point", "fcl")
    """
    stem = filename.replace('.csv', '')
    parts = stem.rsplit('-', 1)  # Split from the right (last dash)

    if len(parts) != 2:
        raise ValueError(f"Could not parse filename: {filename}")

    test_name, library = parts
    return test_name, library


def consolidate_module(module_dir: Path, remove_originals: bool = False) -> int:
    """
    Consolidate all CSV files in a module directory.

    Args:
        module_dir: Path to module directory (e.g., results/cut/)
        remove_originals: If True, remove individual library CSV files after consolidation

    Returns:
        Number of consolidated files created
    """
    # Find all CSV files, excluding already-consolidated ones
    csv_files = [
        f for f in module_dir.glob("*.csv")
        if not f.name.startswith("consolidated-")
    ]

    if not csv_files:
        return 0

    # Group files by test name
    tests: Dict[str, List[Tuple[Path, str]]] = {}

    for csv_file in csv_files:
        try:
            test_name, library = parse_filename(csv_file.name)

            if test_name not in tests:
                tests[test_name] = []

            tests[test_name].append((csv_file, library))
        except ValueError as e:
            print(f"WARNING: Skipping file: {e}")
            continue

    # Consolidate each test group
    consolidated_count = 0

    for test_name, files in tests.items():
        print(f"Consolidating {test_name} ({len(files)} libraries)...")

        # Read all CSVs and add library column
        dfs = []
        for csv_path, library in files:
            try:
                df = pd.read_csv(csv_path)
                df['library'] = library
                dfs.append(df)
            except Exception as e:
                print(f"  ERROR reading {csv_path.name}: {e}")
                continue

        if not dfs:
            print(f"  ERROR: No valid CSVs found for {test_name}")
            continue

        # Concatenate all dataframes
        consolidated_df = pd.concat(dfs, ignore_index=True)

        # Write consolidated CSV with prefix
        output_path = module_dir / f"consolidated-{test_name}.csv"
        consolidated_df.to_csv(output_path, index=False)
        print(f"  Written: {output_path.name} ({len(consolidated_df)} rows)")

        # Optionally remove original files
        if remove_originals:
            for csv_path, _ in files:
                csv_path.unlink()
                print(f"  Removed: {csv_path.name}")

        consolidated_count += 1

    return consolidated_count


################################################################################
# JSON Export
################################################################################

# Mapping from consolidated CSV names to JSON output names
JSON_OUTPUT_NAMES = {
    "boolean": "boolean",
    "embedded_isocurves": "embedded_isocurves",
    "embedded_self_intersection_curves": "self_intersection",
    "isocontours": "isocontours",
    "mesh_mesh_curves": "mesh_mesh_curves",
    "connected_components": "connected_components",
    "boundary_paths": "boundary_paths",
    "point_cloud-build_tree": "point_cloud_build_tree",
    "point_cloud-knn": "point_cloud_knn",
    "polygons-build_tree": "polygons_build_tree",
    "polygons-closest_point": "polygons_closest_point",
    "polygons_to_polygons-closest_point": "mesh_mesh_closest_point",
    "polygons_to_polygons-collision": "mesh_mesh_collision",
    "mod_tree-update": "mod_tree_update",
}


def pivot_simple(df: pd.DataFrame, size_col: str = "polygons") -> pd.DataFrame:
    """Pivot simple benchmarks: group by size, libraries become columns."""
    # Handle duplicate size columns (e.g., polygons0, polygons1)
    if size_col not in df.columns:
        for col in df.columns:
            if col.startswith(size_col) or col == "points":
                size_col = col
                break

    pivot = df.pivot_table(
        index=size_col,
        columns="library",
        values="time_ms",
        aggfunc="first"
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(columns={size_col: size_col.rstrip("0").rstrip("1")})

    # Ensure 'tf' comes first if present
    cols = list(pivot.columns)
    if "tf" in cols:
        cols.remove("tf")
        cols.insert(1, "tf")
        pivot = pivot[cols]

    return pivot


# Libraries that support multiple BV types - append BV suffix
MULTI_BV_LIBRARIES = {"tf", "fcl"}


def pivot_with_bv(df: pd.DataFrame, size_col: str = "polygons") -> pd.DataFrame:
    """Pivot benchmarks with bounding volume: creates columns like tf_aabb, fcl_obbrss.

    For libraries with only one BV type (nanoflann, cgal), uses just the library name.
    """
    df = df.copy()

    # Create column name: append BV suffix only for multi-BV libraries
    def make_col_name(row):
        if row["library"] in MULTI_BV_LIBRARIES:
            return row["library"] + "_" + row["bv"].lower()
        return row["library"]

    df["lib_bv"] = df.apply(make_col_name, axis=1)

    pivot = df.pivot_table(
        index=size_col,
        columns="lib_bv",
        values="time_ms",
        aggfunc="first"
    ).reset_index()

    pivot.columns.name = None

    # Sort columns: size first, then tf variants, then others
    cols = [size_col]
    tf_cols = sorted([c for c in pivot.columns if c.startswith("tf_")])
    other_cols = sorted([c for c in pivot.columns if c != size_col and not c.startswith("tf_")])
    pivot = pivot[cols + tf_cols + other_cols]

    return pivot


def pivot_with_bv_and_extra(df: pd.DataFrame, size_col: str, extra_col: str) -> pd.DataFrame:
    """Pivot benchmarks with BV and extra dimension (e.g., k for KNN).

    For libraries with only one BV type (nanoflann, cgal), uses just the library name.
    """
    df = df.copy()

    # Create column name: append BV suffix only for multi-BV libraries
    def make_col_name(row):
        if row["library"] in MULTI_BV_LIBRARIES:
            return row["library"] + "_" + row["bv"].lower()
        return row["library"]

    df["lib_bv"] = df.apply(make_col_name, axis=1)

    pivot = df.pivot_table(
        index=[size_col, extra_col],
        columns="lib_bv",
        values="time_ms",
        aggfunc="first"
    ).reset_index()

    pivot.columns.name = None

    # Sort columns
    cols = [size_col, extra_col]
    tf_cols = sorted([c for c in pivot.columns if c.startswith("tf_")])
    other_cols = sorted([c for c in pivot.columns if c not in cols and not c.startswith("tf_")])
    pivot = pivot[cols + tf_cols + other_cols]

    return pivot


def pivot_with_extra(df: pd.DataFrame, size_col: str, extra_col: str) -> pd.DataFrame:
    """Pivot benchmarks with extra dimension (e.g., n_cuts for isocontours)."""
    pivot = df.pivot_table(
        index=[size_col, extra_col],
        columns="library",
        values="time_ms",
        aggfunc="first"
    ).reset_index()

    pivot.columns.name = None

    # Ensure 'tf' comes first
    cols = [size_col, extra_col]
    if "tf" in pivot.columns:
        cols.append("tf")
    other_cols = [c for c in pivot.columns if c not in cols]
    pivot = pivot[cols + sorted(other_cols)]

    return pivot


def pivot_mod_tree(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot mod_tree update benchmark: polygons × dirty_pct with update_pct values."""
    pivot = df.pivot_table(
        index="dirty_pct",
        columns="polygons",
        values="update_pct",
        aggfunc="first"
    ).reset_index()

    pivot.columns.name = None
    return pivot


def convert_to_json(csv_path: Path, output_dir: Path) -> Optional[Path]:
    """Convert a consolidated CSV to JSON for charts."""
    test_name = csv_path.stem.replace("consolidated-", "")

    if test_name not in JSON_OUTPUT_NAMES:
        print(f"  SKIP: No JSON mapping for {test_name}")
        return None

    json_name = JSON_OUTPUT_NAMES[test_name]
    output_path = output_dir / f"{json_name}.json"

    df = pd.read_csv(csv_path)

    # Determine pivot strategy based on columns
    has_bv = "bv" in df.columns
    has_k = "k" in df.columns
    has_n_cuts = "n_cuts" in df.columns
    has_dirty_pct = "dirty_pct" in df.columns

    if has_dirty_pct:
        pivot = pivot_mod_tree(df)
    elif has_bv and has_k:
        pivot = pivot_with_bv_and_extra(df, "points", "k")
    elif has_bv:
        size_col = "points" if "points" in df.columns else "polygons"
        pivot = pivot_with_bv(df, size_col)
    elif has_n_cuts:
        pivot = pivot_with_extra(df, "polygons", "n_cuts")
    else:
        pivot = pivot_simple(df)

    # Round numeric columns
    for col in pivot.columns:
        if pivot[col].dtype in ["float64", "float32"]:
            # Keep more precision for very small values (microseconds)
            if pivot[col].max() < 0.01:
                pivot[col] = pivot[col].round(5)
            else:
                pivot[col] = pivot[col].round(2)

    # Convert to JSON
    records = pivot.to_dict(orient="records")

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"  -> {output_path.name} ({len(records)} rows)")
    return output_path


def export_json(results_dir: Path, output_dir: Path) -> int:
    """Export all consolidated CSVs to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for module_dir in sorted(results_dir.iterdir()):
        if not module_dir.is_dir():
            continue

        print(f"Module: {module_dir.name}")

        for csv_path in sorted(module_dir.glob("consolidated-*.csv")):
            result = convert_to_json(csv_path, output_dir)
            if result:
                count += 1

        print()

    return count


################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate benchmark results from per-library to per-test CSVs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)"
    )
    parser.add_argument(
        "--remove-originals",
        action="store_true",
        help="Remove individual library CSV files after consolidation"
    )
    parser.add_argument(
        "--json",
        type=str,
        metavar="OUTPUT_DIR",
        help="Export consolidated CSVs to JSON for charts (e.g., ../docs/benchmarks)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    if not results_dir.is_dir():
        print(f"ERROR: Not a directory: {results_dir}")
        sys.exit(1)

    # JSON export only
    if args.json:
        print(f"Exporting JSON to: {Path(args.json).absolute()}")
        print()
        count = export_json(results_dir, Path(args.json))
        print(f"Done! Exported {count} JSON files")
        return

    # Normal consolidation
    print(f"Consolidating results in: {results_dir.absolute()}")
    print()

    total_consolidated = 0

    # Process each module subdirectory
    for module_dir in sorted(results_dir.iterdir()):
        if not module_dir.is_dir():
            continue

        print(f"Module: {module_dir.name}")
        count = consolidate_module(module_dir, args.remove_originals)
        total_consolidated += count
        print()

    print(f"Done! Consolidated {total_consolidated} test groups")


if __name__ == "__main__":
    main()
