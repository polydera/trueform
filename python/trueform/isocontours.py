"""
Isocontour extraction from scalar fields

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union, Tuple
from . import _trueform
from ._core import Mesh, OffsetBlockedArray


def isocontours(
    mesh: Mesh,
    scalar_field: np.ndarray,
    threshold: Union[float, np.ndarray]
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Extract isocontour curves from a scalar field on a mesh.

    Computes curves where the scalar field crosses the specified threshold value(s).
    Returns paths (as indices into points) and the curve point coordinates.

    Parameters
    ----------
    mesh : Mesh
        The mesh on which to compute isocontours
    scalar_field : np.ndarray
        Scalar values at mesh vertices, shape (num_points,)
        Must have same dtype as mesh (float32 or float64)
    threshold : float or array-like
        Single threshold value or array of multiple threshold values
        If array, computes isocontours for all values efficiently

    Returns
    -------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one curve.
        Iterate over paths to get individual curves: `for path_ids in paths: ...`
    points : np.ndarray
        Curve point coordinates with shape (N, dims)
        Access curve points via: `curve_points = points[paths[i]]`

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Load mesh and create scalar field
    >>> faces, points = tf.read_stl("mesh.stl")
    >>> mesh = tf.Mesh(faces, points)
    >>> plane = tf.Plane(normal=[0.0, 0.0, 1.0], offset=0.0)
    >>> distances = tf.distance_field(mesh.points, plane)
    >>>
    >>> # Extract single isocontour at z=0
    >>> paths, points = tf.isocontours(mesh, distances, 0.0)
    >>> print(f"Found {len(paths)} curve(s)")
    >>>
    >>> # Extract multiple isocontours
    >>> paths, points = tf.isocontours(mesh, distances, [0.0, 0.5, 1.0])
    >>>
    >>> # Iterate over curves
    >>> for path_ids in paths:
    ...     curve_points = points[path_ids]
    ...     # Process curve (e.g., plot, analyze, etc.)
    """

    # Validate mesh
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Expected Mesh, got {type(mesh)}")

    # Validate scalar_field
    if not isinstance(scalar_field, np.ndarray):
        raise TypeError(
            f"Expected numpy array for scalar_field, got {type(scalar_field)}"
        )

    if scalar_field.ndim != 1:
        raise ValueError(
            f"Expected 1D array for scalar_field, got shape {scalar_field.shape}"
        )

    if len(scalar_field) != mesh.number_of_points:
        raise ValueError(
            f"Scalar field size ({len(scalar_field)}) must match number of mesh points ({mesh.number_of_points})"
        )

    # Validate dtype matches mesh
    if scalar_field.dtype != mesh.dtype:
        raise TypeError(
            f"Scalar field dtype ({scalar_field.dtype}) must match mesh dtype ({mesh.dtype})"
        )

    # Ensure C-contiguous
    if not scalar_field.flags['C_CONTIGUOUS']:
        scalar_field = np.ascontiguousarray(scalar_field)

    # Convert threshold to array
    threshold_array = np.atleast_1d(threshold)

    # Validate threshold dtype matches mesh
    if threshold_array.dtype != mesh.dtype:
        # Try to convert to mesh dtype
        threshold_array = threshold_array.astype(mesh.dtype)

    # Ensure C-contiguous
    if not threshold_array.flags['C_CONTIGUOUS']:
        threshold_array = np.ascontiguousarray(threshold_array)

    # Get variant suffix
    index_str = 'int' if mesh.faces.dtype == np.int32 else 'int64'
    real_str = 'float' if mesh.dtype == np.float32 else 'double'
    suffix = f"{index_str}{real_str}{mesh.ngon}{mesh.dims}d"

    # Dispatch to C++ based on threshold count
    if threshold_array.size == 1:
        # Single threshold - use scalar value
        func_name = f"make_isocontours_single_{suffix}"
        threshold_value = float(threshold_array[0])
        paths_offsets, paths_data, points = getattr(_trueform, func_name)(
            mesh._wrapper, scalar_field, threshold_value
        )
    else:
        # Multiple thresholds - use array
        func_name = f"make_isocontours_multi_{suffix}"
        paths_offsets, paths_data, points = getattr(_trueform, func_name)(
            mesh._wrapper, scalar_field, threshold_array
        )

    # Wrap paths in OffsetBlockedArray
    paths = OffsetBlockedArray(paths_offsets, paths_data)

    return paths, points
