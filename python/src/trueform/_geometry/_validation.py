"""
Validation utilities for alignment inputs.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np


def validate_alignment_normals(normals, cloud, name):
    """
    Validate a (cloud, normals) tuple's normals against its cloud.

    Parameters
    ----------
    normals : np.ndarray
        Per-point normals riding the cloud
    cloud : PointCloud
        The cloud whose points the normals annotate
    name : str
        Name for error messages ("normals0" or "normals1")

    Returns
    -------
    np.ndarray
        Validated, C-contiguous normals in the cloud's dtype

    Raises
    ------
    TypeError
        If normals is not a numpy array
    ValueError
        If normals is not (N, 3) or its row count differs from the cloud's
        point count
    """
    if not isinstance(normals, np.ndarray):
        raise TypeError(f"Expected numpy array for {name}, got {type(normals)}")
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {normals.shape}")
    if normals.shape[0] != cloud.size:
        raise ValueError(
            f"Normals count mismatch: {name} has {normals.shape[0]} rows, "
            f"its cloud has {cloud.size} points"
        )
    if normals.dtype != cloud.dtype or not normals.flags["C_CONTIGUOUS"]:
        normals = np.ascontiguousarray(normals, dtype=cloud.dtype)
    return normals
