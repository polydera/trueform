"""
Face normals computation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._primitives import Triangle, Polygon
from .._primitives.primitive import Primitive
from .._core import OffsetBlockedArray
from .._dispatch import ensure_mesh, InputMeta, build_suffix


def normals(
    data: Union[Triangle, Polygon, Mesh, Tuple[np.ndarray, np.ndarray],
                Tuple[OffsetBlockedArray, np.ndarray]]
) -> np.ndarray:
    """
    Compute face normals for a triangle, polygon, or mesh.

    Parameters
    ----------
    data : Triangle, Polygon, Mesh, or tuple
        - Triangle: Single 3D triangle returns (3,) unit normal;
          batch of N triangles returns (N, 3) unit normals.
        - Polygon: Single 3D polygon returns (3,) unit normal;
          batch of N polygons returns (N, 3) unit normals.
        - Mesh: Mesh object (must be 3D)
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    normals : np.ndarray
        Unit face normals. Shape (3,) for single primitive, (N, 3) for batch,
        or (num_faces, 3) for mesh.

    Raises
    ------
    TypeError
        If data is not a Triangle, Polygon, Mesh, or valid tuple.
    ValueError
        If data is not 3D.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Single triangle
    >>> tri = tf.Triangle(a=[0,0,0], b=[1,0,0], c=[0,1,0])
    >>> tf.normals(tri)
    array([0., 0., 1.], dtype=float32)
    >>>
    >>> # Batch of triangles
    >>> tris = tf.Triangle(np.random.rand(50, 3, 3).astype(np.float32))
    >>> n = tf.normals(tris)  # shape (50, 3)
    >>>
    >>> # From Mesh
    >>> mesh = tf.Mesh(faces, points)
    >>> n = tf.normals(mesh)
    >>>
    >>> # From tuple
    >>> n = tf.normals((faces, points))
    """
    # Triangle or Polygon primitive → dispatch to C++ primitive binding
    if isinstance(data, (Triangle, Polygon)):
        if data.dims != 3:
            raise ValueError(
                f"normals() requires 3D primitives, got {data.dims}D")
        suffix = build_suffix(InputMeta(None, data.dtype, None, data.dims))
        fn = getattr(_trueform.spatial, f"normals_prim_{suffix}")
        return fn(data._wrapper)

    mesh = ensure_mesh(data, dims=3)
    return mesh.normals
