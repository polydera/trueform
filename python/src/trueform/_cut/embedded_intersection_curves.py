"""
Embed intersection curves between two meshes into first mesh

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple, Union
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import extract_meta, build_suffix_pair


def embedded_intersection_curves(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]]:
    """
    Embed intersection curves between mesh A and mesh B into mesh A.

    Splits faces of mesh0 along intersection curves with mesh1, so that the
    intersections become edges in the resulting mesh. All faces from mesh0
    are preserved (split where intersecting), with no faces from mesh1.
    Supports both triangle meshes and dynamic (variable polygon size) meshes.

    Parameters
    ----------
    mesh0 : Mesh
        3D mesh to embed curves into (triangle or dynamic)
    mesh1 : Mesh
        3D mesh providing the cutting surface (triangle or dynamic)
        Must have same real dtype (float32 or float64) as mesh0
    return_curves : bool, default False
        If True, also return the intersection curves

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the result mesh. Returns np.ndarray with shape (N, 3)
        if both inputs are triangle meshes, otherwise OffsetBlockedArray.
    result_points : np.ndarray
        Point coordinates of the result mesh, shape (M, 3)
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True
        Intersection curves as indices into curve_points
    curve_points : np.ndarray, optional
        Only returned if return_curves=True
        Curve point coordinates with shape (P, 3)

    Raises
    ------
    ValueError
        If meshes are not 3D or have mismatched dtypes
    TypeError
        If inputs are not Mesh objects

    Examples
    --------
    >>> import trueform as tf
    >>> # Load two meshes
    >>> mesh0 = tf.Mesh(*tf.read_stl("mesh0.stl"))
    >>> mesh1 = tf.Mesh(*tf.read_stl("mesh1.stl"))
    >>>
    >>> # Embed intersection curves into mesh0
    >>> faces, points = tf.embedded_intersection_curves(mesh0, mesh1)
    >>> print(f"Result has {len(faces)} faces")
    >>>
    >>> # Get curves as well
    >>> (faces, points), (paths, curve_pts) = tf.embedded_intersection_curves(
    ...     mesh0, mesh1, return_curves=True
    ... )
    >>> print(f"Found {len(paths)} intersection curve(s)")

    Notes
    -----
    This is useful when you need to mark where meshes intersect without
    carving—for example, projecting cutting guides onto a surface or
    visualizing contact regions.

    Unlike boolean operations, this function does not remove any faces
    from mesh0. The output mesh has the same volume and surface area
    as mesh0, with additional edges along the intersection curves.
    """

    if not isinstance(mesh0, Mesh):
        raise TypeError(
            f"mesh0 must be a Mesh object, got {type(mesh0).__name__}. "
            f"Topology information is required for embedded_intersection_curves."
        )

    if not isinstance(mesh1, Mesh):
        raise TypeError(
            f"mesh1 must be a Mesh object, got {type(mesh1).__name__}. "
            f"Topology information is required for embedded_intersection_curves."
        )

    if mesh0.dims != 3:
        raise ValueError(
            f"embedded_intersection_curves only supports 3D meshes, got mesh0 with {mesh0.dims}D"
        )
    if mesh1.dims != 3:
        raise ValueError(
            f"embedded_intersection_curves only supports 3D meshes, got mesh1 with {mesh1.dims}D"
        )

    if mesh0.ngon != 3 and not mesh0.is_dynamic:
        raise ValueError(
            f"embedded_intersection_curves only supports triangle or dynamic meshes, got mesh0 with {mesh0.ngon}-gons"
        )
    if mesh1.ngon != 3 and not mesh1.is_dynamic:
        raise ValueError(
            f"embedded_intersection_curves only supports triangle or dynamic meshes, got mesh1 with {mesh1.ngon}-gons"
        )

    if mesh0.dtype != mesh1.dtype:
        raise ValueError(
            f"Mesh dtypes must match: mesh0 has {mesh0.dtype}, mesh1 has {mesh1.dtype}. "
            f"Convert both meshes to the same dtype (float32 or float64)."
        )

    # Unlike boolean, we don't canonicalize index order because
    # embedded_intersection_curves is asymmetric (always embeds into mesh0)
    meta0 = extract_meta(mesh0)
    meta1 = extract_meta(mesh1)
    suffix = build_suffix_pair(meta0, meta1)

    # Only mesh0 faces are returned, so result type depends only on mesh0
    result_is_dynamic = mesh0.is_dynamic

    if return_curves:
        func_name = f"embedded_intersection_curves_curves_mesh_mesh_{suffix}"
        (result_faces, result_points), ((paths_offsets, paths_data), curve_points) = getattr(
            _trueform.cut, func_name
        )(mesh0._wrapper, mesh1._wrapper)

        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        paths = OffsetBlockedArray(paths_offsets, paths_data)

        return (result_faces, result_points), (paths, curve_points)
    else:
        func_name = f"embedded_intersection_curves_mesh_mesh_{suffix}"
        result_faces, result_points = getattr(_trueform.cut, func_name)(
            mesh0._wrapper, mesh1._wrapper
        )

        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        return result_faces, result_points
