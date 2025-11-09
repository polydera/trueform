"""
IO utilities for reading and writing mesh files

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Tuple
from . import _trueform


def read_stl(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an STL file and return mesh data as numpy arrays.

    Parameters
    ----------
    filename : str
        Path to the STL file to read

    Returns
    -------
    faces : ndarray of shape (num_faces, 3) and dtype int32
        Face indices into the points array. Each row contains three indices
        that reference vertices in the points array, forming a triangle.
    points : ndarray of shape (num_points, 3) and dtype float32
        3D coordinates of mesh vertices. Each row is a (x, y, z) coordinate.

    Examples
    --------
    >>> import trueform as tf
    >>> faces, points = tf.read_stl("model.stl")
    >>> print(f"Mesh has {len(faces)} faces and {len(points)} vertices")
    >>> print(f"First face indices: {faces[0]}")
    >>> print(f"First vertex: {points[0]}")

    Notes
    -----
    The STL file format stores triangular mesh data. This function:
    - Reads both ASCII and binary STL files
    - Cleans duplicate vertices (merges points that are identical)
    - Returns zero-copy numpy arrays (memory is managed by numpy)
    - Face indices are 0-based
    """
    return _trueform.read_stl(filename)
