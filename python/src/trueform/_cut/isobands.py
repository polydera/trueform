"""
Isoband extraction from scalar fields

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple, Optional
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix


def isobands(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh],
    scalar_field: np.ndarray,
    cut_values: Union[float, np.ndarray],
    selected_bands: Optional[np.ndarray] = None,
    return_curves: bool = False
) -> Union[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
           Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[OffsetBlockedArray, np.ndarray]]]:
    """
    Extract isobands from a scalar field on a 3D mesh.

    Computes bands (regions) between consecutive threshold values.
    Returns band geometry as (faces, points) and labels indicating which band each face belongs to.

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
    cut_values : float or array-like
        Threshold values defining band boundaries
        If array, creates bands between consecutive values
    selected_bands : array-like of int, optional
        Indices of bands to extract. If None, extracts all bands.
        Bands are numbered 0 to len(cut_values), where band i is the region
        between cut_values[i-1] and cut_values[i] (with bands 0 and len(cut_values)
        being unbounded on one side)
    return_curves : bool, default False
        If True, also return the boundary curves between bands

    Returns
    -------
    faces : np.ndarray or OffsetBlockedArray
        Face indices of the band geometry, shape (num_faces, 3) for triangles,
        or OffsetBlockedArray for dynamic meshes
    points : np.ndarray
        Point coordinates of the band geometry, shape (num_points, 3)
    labels : np.ndarray
        Band labels for each face in the output mesh, shape (num_faces,)
        Values correspond to indices in selected_bands
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True
        Boundary curves as indices into curve_points
    curve_points : np.ndarray, optional
        Only returned if return_curves=True
        Curve point coordinates with shape (N, 3)

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
    >>> # Extract isobands at different height levels using Mesh
    >>> (band_faces, band_points), labels = tf.isobands(mesh, distances, [-1.0, 0.0, 1.0])
    >>> print(f"Created {len(band_faces)} faces in {len(np.unique(labels))} bands")
    >>>
    >>> # Extract using tuple input
    >>> (band_faces, band_points), labels = tf.isobands((faces, points), distances, [-1.0, 0.0, 1.0])
    >>>
    >>> # Extract only specific bands
    >>> (band_faces, band_points), labels = tf.isobands(
    ...     mesh, distances, [-1.0, 0.0, 1.0], selected_bands=[0, 2]
    ... )
    >>>
    >>> # Extract with boundary curves
    >>> (band_faces, band_points), labels, (paths, curve_points) = tf.isobands(
    ...     mesh, distances, [-1.0, 0.0, 1.0], return_curves=True
    ... )
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
        raise ValueError(f"isobands only supports 3D meshes, got {mesh.dims}D")

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

    # Convert cut_values to array
    cut_values_array = np.atleast_1d(cut_values)

    # Validate cut_values dtype matches mesh
    if cut_values_array.dtype != mesh.dtype:
        # Try to convert to mesh dtype
        cut_values_array = cut_values_array.astype(mesh.dtype)

    # Ensure C-contiguous
    if not cut_values_array.flags['C_CONTIGUOUS']:
        cut_values_array = np.ascontiguousarray(cut_values_array)

    # Handle selected_bands
    if selected_bands is None:
        # Default: all bands [0, 1, ..., len(cut_values)]
        selected_bands_array = np.arange(len(cut_values_array) + 1, dtype=np.int32)
    else:
        selected_bands_array = np.atleast_1d(selected_bands).astype(np.int32)

    # Ensure C-contiguous
    if not selected_bands_array.flags['C_CONTIGUOUS']:
        selected_bands_array = np.ascontiguousarray(selected_bands_array)

    # Get variant suffix
    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, mesh.dims)
    suffix = build_suffix(meta)

    # Dispatch to C++ based on return_curves
    if return_curves:
        func_name = f"make_isobands_curves_{suffix}"
        (result_faces, result_points), labels, ((paths_offsets, paths_data), curve_points) = getattr(
            _trueform.cut, func_name
        )(mesh._wrapper, scalar_field, cut_values_array, selected_bands_array)

        # Wrap result faces in OffsetBlockedArray if dynamic
        if mesh.is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        # Wrap paths in OffsetBlockedArray
        paths = OffsetBlockedArray(paths_offsets, paths_data)

        return (result_faces, result_points), labels, (paths, curve_points)
    else:
        func_name = f"make_isobands_{suffix}"
        (result_faces, result_points), labels = getattr(_trueform.cut, func_name)(
            mesh._wrapper, scalar_field, cut_values_array, selected_bands_array
        )

        # Wrap result faces in OffsetBlockedArray if dynamic
        if mesh.is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        return (result_faces, result_points), labels
