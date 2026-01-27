"""
Command-line argument parsing for VTK examples.

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import argparse


def create_parser(description: str, mesh_args: int | str = 1) -> argparse.ArgumentParser:
    """
    Create argument parser for VTK examples.

    Parameters
    ----------
    description : str
        Description shown in --help
    mesh_args : int or str
        1 = single mesh, 2 = two meshes (second optional), "many" = one or more meshes

    Returns
    -------
    argparse.ArgumentParser
        Parser with mesh arguments configured. Add epilog for controls before calling parse_args().
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    if mesh_args == 1:
        parser.add_argument("mesh", help="Path to mesh file (STL)")
    elif mesh_args == 2:
        parser.add_argument("mesh1", help="Path to first mesh file")
        parser.add_argument(
            "mesh2", nargs="?", default=None,
            help="Path to second mesh (optional, uses mesh1 if omitted)"
        )
    elif mesh_args == "many":
        parser.add_argument("meshes", nargs="+", help="Path(s) to mesh files")

    return parser
