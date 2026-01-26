"""
Embed self-intersection curves into mesh topology

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple, Union
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix


def embedded_self_intersection_curves(
    mesh: Mesh,
    return_curves: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]]:
    """
    Embed self-intersection curves into mesh topology.

    Splits faces along self-intersection curves so that the intersections
    become edges in the resulting mesh. This is useful for mesh repair
    workflows where self-intersections need to be resolved.
    Supports both triangle meshes and dynamic (variable polygon size) meshes.

    Parameters
    ----------
    mesh : Mesh
        3D mesh with potential self-intersections (triangle or dynamic)
    return_curves : bool, default False
        If True, also return the self-intersection curves

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the result mesh. Returns np.ndarray with shape (N, 3)
        if input is triangle mesh, otherwise OffsetBlockedArray.
    result_points : np.ndarray
        Point coordinates of the result mesh, shape (M, 3)
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True
        Self-intersection curves as indices into curve_points
    curve_points : np.ndarray, optional
        Only returned if return_curves=True
        Curve point coordinates with shape (P, 3)

    Raises
    ------
    ValueError
        If mesh is not 3D
    TypeError
        If input is not a Mesh object

    Examples
    --------
    >>> import trueform as tf
    >>> # Load a mesh with self-intersections
    >>> faces, points = tf.read_stl("self_intersecting.stl")
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> # Embed self-intersection curves into mesh
    >>> result_faces, result_points = tf.embedded_self_intersection_curves(mesh)
    >>> print(f"Result has {len(result_faces)} faces")
    >>>
    >>> # Get curves as well
    >>> (faces, points), (paths, curve_pts) = tf.embedded_self_intersection_curves(
    ...     mesh, return_curves=True
    ... )
    >>> print(f"Found {len(paths)} self-intersection curve(s)")

    Notes
    -----
    The input mesh remains unchanged. The output mesh has faces split such
    that no face contains a self-intersection curve in its interior - all
    self-intersections become edges.

    This operation is useful for:
    - Mesh repair pipelines
    - Preparing meshes for boolean operations
    - Visualization of self-intersection regions
    """

    # 1. VALIDATE INPUT IS MESH OBJECT
    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be a Mesh object, got {type(mesh).__name__}. "
            f"Topology information is required for embedded_self_intersection_curves."
        )

    # 2. VALIDATE 3D
    if mesh.dims != 3:
        raise ValueError(
            f"embedded_self_intersection_curves only supports 3D meshes, got mesh with {mesh.dims}D"
        )

    # 3. VALIDATE TRIANGLES OR DYNAMIC
    if mesh.ngon != 3 and not mesh.is_dynamic:
        raise ValueError(
            f"embedded_self_intersection_curves only supports triangle or dynamic meshes, got ngon={mesh.ngon}"
        )

    # 4. BUILD SUFFIX FOR C++ FUNCTION
    ngon = 'dyn' if mesh.is_dynamic else '3'
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    # 5. DISPATCH TO C++
    if return_curves:
        func_name = f"embedded_self_intersection_curves_curves_mesh_{suffix}"
        (result_faces, result_points), ((paths_offsets, paths_data), curve_points) = getattr(
            _trueform.cut, func_name
        )(mesh._wrapper)

        # Wrap result faces in OffsetBlockedArray if dynamic
        if mesh.is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        # Wrap paths in OffsetBlockedArray
        paths = OffsetBlockedArray(paths_offsets, paths_data)

        return (result_faces, result_points), (paths, curve_points)
    else:
        func_name = f"embedded_self_intersection_curves_mesh_{suffix}"
        result_faces, result_points = getattr(_trueform.cut, func_name)(
            mesh._wrapper
        )

        # Wrap result faces in OffsetBlockedArray if dynamic
        if mesh.is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        return result_faces, result_points
