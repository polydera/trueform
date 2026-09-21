"""
Per-face quality measures

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import ensure_mesh, extract_meta, build_suffix


def face_quality(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]
):
    """
    Measure every face of a mesh.

    The corner angles and the aspect ratio are stated for a face of any
    arity; the quality is the triangle measure, and a face that is not
    a triangle has none and reads -1.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: Mesh object
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    quality : np.ndarray of shape (num_faces,)
        Triangle quality: 1 for equilateral, approaching 0 for a
        sliver; -1 for a face that is not a triangle.
    min_angle : np.ndarray of shape (num_faces,)
        Smallest corner angle, in radians. An angle is unsigned and
        lies in [0, pi], so a reflex corner of a non-convex face reads
        its explement.
    max_angle : np.ndarray of shape (num_faces,)
        Largest corner angle, in radians.
    aspect_ratio : np.ndarray of shape (num_faces,)
        The longest side over the shortest, infinite where a side has
        no length.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]], dtype=np.float32
    ... )
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> quality, min_angle, max_angle, aspect_ratio = tf.face_quality(mesh)
    >>> quality
    array([1.], dtype=float32)
    """
    mesh = ensure_mesh(data, dims=3)
    meta = extract_meta(mesh)
    suffix = build_suffix(meta)
    func = getattr(_trueform.geometry, f"face_quality_{suffix}")
    return func(mesh._wrapper)
