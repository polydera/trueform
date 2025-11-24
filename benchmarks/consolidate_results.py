#!/usr/bin/env python3
"""
Consolidate benchmark results by merging per-library CSV files into unified per-test files.

Example:
    results/cut/boolean-tf.csv
    results/cut/boolean-cgal.csv
    results/cut/boolean-igl.csv

    -> results/cut/consolidated-boolean.csv (with 'library' column: tf, cgal, igl)

Consolidated files are prefixed with 'consolidated-' and are skipped in subsequent runs.

Usage:
    python consolidate_results.py
    python consolidate_results.py --remove-originals
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
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
        if len(files) == 1:
            # Only one library for this test, skip consolidation
            print(f"Skipping {test_name}: only one library variant")
            continue

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

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    if not results_dir.is_dir():
        print(f"ERROR: Not a directory: {results_dir}")
        sys.exit(1)

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
