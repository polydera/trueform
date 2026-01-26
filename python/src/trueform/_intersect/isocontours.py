"""
Isocontour extraction from scalar fields

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix


def isocontours(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh],
    scalar_field: np.ndarray,
    threshold: Union[float, np.ndarray]
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Extract isocontour curves from a scalar field on a mesh.

    Computes curves where the scalar field crosses the specified threshold value(s).
    Returns paths (as indices into points) and the curve point coordinates.

    Parameters
    ----------
    data : tuple or Mesh
        Input mesh data:
        - Tuple (faces, points) where:
          * faces: shape (N, 3) with dtype int32 or int64, or OffsetBlockedArray for dynamic
          * points: shape (M, 3) with dtype float32 or float64
        - Mesh object (must be 3D triangular or dynamic mesh)
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
    >>> # Extract single isocontour at z=0 using Mesh
    >>> paths, points = tf.isocontours(mesh, distances, 0.0)
    >>> print(f"Found {len(paths)} curve(s)")
    >>>
    >>> # Extract using tuple input
    >>> paths, points = tf.isocontours((faces, points), distances, 0.0)
    >>>
    >>> # Extract multiple isocontours
    >>> paths, points = tf.isocontours(mesh, distances, [0.0, 0.5, 1.0])
    >>>
    >>> # Iterate over curves
    >>> for path_ids in paths:
    ...     curve_points = points[path_ids]
    ...     # Process curve (e.g., plot, analyze, etc.)
    """

    # Normalize input to Mesh object
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (faces, points), got {len(data)}"
            )
        faces, points = data
        mesh = Mesh(faces, points)
    elif isinstance(data, Mesh):
        mesh = data
    else:
        raise TypeError(
            f"Expected Mesh or (faces, points) tuple, got {type(data).__name__}"
        )

    # Only support 3D meshes
    if mesh.dims != 3:
        raise ValueError(f"isocontours only supports 3D meshes, got {mesh.dims}D")

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
    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, mesh.dims)
    suffix = build_suffix(meta)

    # Dispatch to C++ based on threshold count
    if threshold_array.size == 1:
        # Single threshold - use scalar value
        func_name = f"make_isocontours_single_{suffix}"
        threshold_value = float(threshold_array[0])
        (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
            mesh._wrapper, scalar_field, threshold_value
        )
    else:
        # Multiple thresholds - use array
        func_name = f"make_isocontours_multi_{suffix}"
        (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
            mesh._wrapper, scalar_field, threshold_array
        )

    # Wrap paths in OffsetBlockedArray
    paths = OffsetBlockedArray(paths_offsets, paths_data)

    return paths, points
