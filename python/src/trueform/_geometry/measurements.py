"""
Mesh measurement functions (volume, area, mean_edge_length)

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._primitives import Polygon, Triangle
from .._core import OffsetBlockedArray
from .._primitives.primitive import Primitive
from .._dispatch import ensure_mesh, extract_meta, build_suffix, InputMeta


def signed_volume(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]
) -> float:
    """
    Compute signed volume of a closed 3D mesh.

    Uses the divergence theorem. Positive volume indicates outward-facing
    normals (CCW winding when viewed from outside), negative indicates
    inward-facing.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: tf.Mesh object (must be 3D)
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    float
        The signed volume.

    Examples
    --------
    >>> import trueform as tf
    >>> faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    >>> tf.signed_volume((faces, points))  # From tuple
    1.0
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.signed_volume(mesh)  # From Mesh
    1.0
    """
    mesh = ensure_mesh(data, dims=3)
    suffix = build_suffix(extract_meta(mesh))
    func = getattr(_trueform.geometry, f"signed_volume_{suffix}")
    return func(mesh._wrapper)


def volume(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]
) -> float:
    """
    Compute volume of a closed 3D mesh.

    Returns the absolute value of the signed volume.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: tf.Mesh object (must be 3D)
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    float
        The volume (always non-negative).

    Examples
    --------
    >>> import trueform as tf
    >>> faces, points = tf.make_box_mesh(2.0, 3.0, 4.0)
    >>> tf.volume((faces, points))
    24.0
    """
    return abs(signed_volume(data))


def area(
    data: Union[np.ndarray, Triangle, Polygon, Mesh, Tuple[np.ndarray, np.ndarray],
                Tuple[OffsetBlockedArray, np.ndarray]]
):
    """
    Compute area of a triangle, polygon, or total surface area of a mesh.

    Parameters
    ----------
    data : Triangle, Polygon, np.ndarray, Mesh, or tuple
        - Triangle: Single or batch of triangles. Returns scalar or (N,) array.
        - Polygon: Single or batch of polygons. Returns scalar or (N,) array.
        - np.ndarray shape (V, D): Single polygon with V vertices in D dimensions
        - Mesh: Triangle mesh (returns total surface area)
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    float or np.ndarray
        Scalar for single primitive or mesh, array for batch primitives.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Single triangle
    >>> tri = tf.Triangle(a=[0,0,0], b=[1,0,0], c=[0,1,0])
    >>> tf.area(tri)
    0.5
    >>>
    >>> # Batch of triangles
    >>> tris = tf.Triangle(np.random.rand(50, 3, 3).astype(np.float32))
    >>> areas = tf.area(tris)  # shape (50,)
    >>>
    >>> # Single polygon
    >>> poly = tf.Polygon([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    >>> tf.area(poly)
    1.0
    >>>
    >>> # Mesh (unit cube surface area = 6)
    >>> faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    >>> tf.area((faces, points))
    6.0
    """
    # Triangle or Polygon primitive → dispatch to C++ primitive binding
    if isinstance(data, (Triangle, Polygon)):
        suffix = build_suffix(InputMeta(None, data.dtype, None, data.dims))
        fn = getattr(_trueform.spatial, f"area_prim_{suffix}")
        return fn(data._wrapper)

    # Handle raw ndarray polygon (no wrapper)
    if isinstance(data, np.ndarray):
        meta = extract_meta(data)
        if meta.index_dtype is None:
            suffix = build_suffix(meta)
            func = getattr(_trueform.geometry, f"area_{suffix}")
            return func(data)

    # Mesh or tuple
    mesh = ensure_mesh(data)
    suffix = build_suffix(extract_meta(mesh))
    func = getattr(_trueform.geometry, f"area_{suffix}")
    return func(mesh._wrapper)


def mean_edge_length(
    data: Union[Triangle, Polygon, Mesh, Tuple[np.ndarray, np.ndarray],
                Tuple[OffsetBlockedArray, np.ndarray]]
) -> float:
    """
    Compute mean edge length of a triangle, polygon, or mesh.

    Parameters
    ----------
    data : Triangle, Polygon, Mesh, or tuple
        - Triangle: Single or batch. Returns mean over all edges.
        - Polygon: Single or batch. Returns mean over all edges.
        - Mesh: tf.Mesh object
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    float
        The mean edge length.

    Examples
    --------
    >>> import trueform as tf
    >>> tri = tf.Triangle(a=[0,0,0], b=[1,0,0], c=[0,1,0])
    >>> tf.mean_edge_length(tri)
    1.1380...
    >>>
    >>> mesh = tf.Mesh(*tf.read_stl("model.stl"))
    >>> tf.mean_edge_length(mesh)
    0.042
    """
    # Triangle or Polygon primitive — pure numpy
    if isinstance(data, (Triangle, Polygon)):
        verts = data.data
        v_next = np.roll(verts, -1, axis=-2)
        edges = (v_next - verts).reshape(-1, verts.shape[-1])
        lengths = np.linalg.norm(edges, axis=1)
        return float(lengths.mean())

    transform = None
    if isinstance(data, Mesh):
        faces, points = data.faces, data.points
        transform = data.transformation
    elif isinstance(data, tuple) and len(data) == 2:
        faces, points = data
    else:
        raise TypeError(
            f"data must be a Triangle, Polygon, Mesh, or (faces, points) tuple, "
            f"got {type(data).__name__}"
        )

    if isinstance(faces, OffsetBlockedArray):
        offsets = faces.offsets
        idx = faces.data
        next_pos = np.arange(len(idx)) + 1
        next_pos[offsets[1:] - 1] = offsets[:-1]
        edges = points[idx[next_pos]] - points[idx]
    else:
        p = points[faces]
        p_next = np.roll(p, -1, axis=1)
        edges = (p_next - p).reshape(-1, points.shape[1])

    if transform is not None:
        rotation = transform[:3, :3]
        edges = edges @ rotation.T

    lengths = np.linalg.norm(edges, axis=1)
    return float(lengths.mean())
