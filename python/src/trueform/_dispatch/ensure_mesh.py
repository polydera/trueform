"""
Ensure input is a Mesh object.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np


def ensure_mesh(data, dims: int = None):
    """
    Ensure input is a Mesh object, wrapping if necessary.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: returned as-is
        - (faces, points): Tuple wrapped in Mesh
        - (OffsetBlockedArray, points): Dynamic mesh wrapped in Mesh
    dims : int, optional
        If provided, validate mesh has this dimensionality (2 or 3).
        Raises ValueError if mismatch.

    Returns
    -------
    Mesh
        The input as a Mesh object.

    Raises
    ------
    TypeError
        If data is not a Mesh or valid tuple.
    ValueError
        If dims is specified and mesh dimensionality doesn't match.

    Examples
    --------
    >>> mesh = ensure_mesh((faces, points))
    >>> mesh = ensure_mesh(existing_mesh)
    >>> mesh = ensure_mesh(data, dims=3)  # Validate 3D
    """
    from .._spatial import Mesh
    from .._core import OffsetBlockedArray

    if isinstance(data, Mesh):
        mesh = data
    elif isinstance(data, tuple) and len(data) == 2:
        faces, points = data
        if isinstance(faces, (np.ndarray, OffsetBlockedArray)):
            mesh = Mesh(faces, points)
        else:
            raise TypeError(
                f"Expected faces to be ndarray or OffsetBlockedArray, "
                f"got {type(faces).__name__}"
            )
    else:
        raise TypeError(
            f"Expected Mesh or (faces, points) tuple, got {type(data).__name__}"
        )

    if dims is not None and mesh.dims != dims:
        raise ValueError(
            f"Expected {dims}D mesh, got {mesh.dims}D"
        )

    return mesh
