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
from .._primitives import Polygon
from .._core import OffsetBlockedArray
from .._dispatch import ensure_mesh, extract_meta, build_suffix


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
    data: Union[np.ndarray, Polygon, Mesh, Tuple[np.ndarray, np.ndarray],
                Tuple[OffsetBlockedArray, np.ndarray]]
) -> float:
    """
    Compute area of a polygon or total surface area of a mesh.

    Parameters
    ----------
    data : np.ndarray, Polygon, Mesh, or tuple
        - np.ndarray shape (N, D): Single polygon with N vertices in D dimensions
        - Polygon: tf.Polygon object
        - Mesh: Triangle mesh (returns total surface area)
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    float
        Area of the polygon, or total surface area of all faces.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Single polygon as array
    >>> polygon = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
    >>> tf.area(polygon)
    1.0
    >>>
    >>> # Single polygon as Polygon object
    >>> poly = tf.Polygon([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    >>> tf.area(poly)
    1.0
    >>>
    >>> # Mesh (unit cube surface area = 6)
    >>> faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    >>> tf.area((faces, points))
    6.0
    """
    # Handle Polygon object - extract vertices
    if isinstance(data, Polygon):
        data = data.vertices

    meta = extract_meta(data)

    # Single polygon (points only, no index dtype means it's just points)
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
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray],
                Tuple[OffsetBlockedArray, np.ndarray]]
) -> float:
    """
    Compute mean edge length of a polygon mesh.

    Parameters
    ----------
    data : Mesh or tuple
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
    >>> mesh = tf.Mesh(*tf.read_stl("model.stl"))
    >>> tf.mean_edge_length(mesh)
    0.042
    """
    transform = None
    if isinstance(data, Mesh):
        faces, points = data.faces, data.points
        transform = data.transformation
    elif isinstance(data, tuple) and len(data) == 2:
        faces, points = data
    else:
        raise TypeError(
            f"data must be a Mesh or (faces, points) tuple, "
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
