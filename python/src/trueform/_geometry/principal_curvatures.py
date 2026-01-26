"""
Principal curvatures computation

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


def principal_curvatures(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]],
    k: int = 2,
    directions: bool = False
):
    """
    Compute principal curvatures at each vertex.

    Principal curvatures (k0, k1) characterize surface curvature at each vertex.
    k0 is the maximum curvature, k1 is the minimum curvature.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: Mesh object
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh
    k : int, default 2
        k-ring neighborhood size for curvature estimation.
    directions : bool, default False
        If True, also return principal directions (d0, d1).

    Returns
    -------
    k0 : np.ndarray of shape (num_points,)
        Maximum principal curvature at each vertex.
    k1 : np.ndarray of shape (num_points,)
        Minimum principal curvature at each vertex.
    d0 : np.ndarray of shape (num_points, 3), optional
        Direction of maximum curvature (only if directions=True).
    d1 : np.ndarray of shape (num_points, 3), optional
        Direction of minimum curvature (only if directions=True).

    Raises
    ------
    TypeError
        If data is not a Mesh or valid tuple.
    ValueError
        If mesh is not 3D.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> faces = np.array([[0,1,2], [1,3,2]], dtype=np.int32)
    >>> points = np.array([[0,0,0],[1,0,0],[0.5,1,0],[1.5,1,0]], dtype=np.float32)
    >>>
    >>> # From Mesh
    >>> mesh = tf.Mesh(faces, points)
    >>> k0, k1 = tf.principal_curvatures(mesh)
    >>>
    >>> # With directions
    >>> k0, k1, d0, d1 = tf.principal_curvatures(mesh, directions=True)
    >>>
    >>> # From tuple with custom k-ring
    >>> k0, k1 = tf.principal_curvatures((faces, points), k=3)
    """
    mesh = ensure_mesh(data, dims=3)
    meta = extract_meta(mesh)
    suffix = build_suffix(meta)
    func = getattr(_trueform.geometry, f"principal_curvatures_{suffix}")
    return func(mesh._wrapper, k, directions)
