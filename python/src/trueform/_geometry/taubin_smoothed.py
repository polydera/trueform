"""
Taubin smoothing implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray
from .._spatial import Mesh
from .._dispatch import topology_suffix


def taubin_smoothed(
    data: Union[Mesh, Tuple[np.ndarray, OffsetBlockedArray]],
    iterations: int = 1,
    lambda_: float = 0.5,
    kpb: float = 0.1
) -> np.ndarray:
    """
    Apply Taubin smoothing to a point set.

    Taubin smoothing alternates between shrinking (positive lambda) and
    inflating (negative mu) passes to smooth without significant volume loss.
    This avoids the shrinkage problem of standard Laplacian smoothing.

    The mu parameter is computed internally from lambda and the pass-band
    frequency kpb: mu = 1 / (kpb - 1/lambda)

    With lambda=0.5 and kpb=0.1, mu ≈ -0.526.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: Mesh object with vertex connectivity
        - (points, vertex_link): Tuple with point coordinates and vertex connectivity

    iterations : int, optional
        Number of smoothing iterations. Each iteration consists of one shrink
        pass followed by one inflate pass. Default is 1.

    lambda_ : float, optional
        Smoothing factor for shrink pass. Default is 0.5.

    kpb : float, optional
        Pass-band frequency. Controls how much high-frequency detail is preserved.
        Lower values preserve more detail. Default is 0.1.

    Returns
    -------
    points : np.ndarray of shape (num_points, dims)
        Smoothed point positions.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> faces = np.array([[0,1,2], [1,3,2]], dtype=np.int32)
    >>> points = np.array([[0,0,0],[1,0,0],[0.5,1,0],[1.5,1,0]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> smoothed = tf.taubin_smoothed(mesh, iterations=10, lambda_=0.5)
    >>>
    >>> # From tuple (points, vertex_link)
    >>> vl = mesh.vertex_link
    >>> smoothed = tf.taubin_smoothed((points, vl), iterations=10)
    """

    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError(f"iterations must be a non-negative integer, got {iterations}")

    if isinstance(data, Mesh):
        points = data.points
        vertex_link = data.vertex_link
    elif isinstance(data, tuple) and len(data) == 2:
        points, vertex_link = data
    else:
        raise TypeError(
            f"data must be Mesh or (points, vertex_link) tuple, "
            f"got {type(data).__name__}"
        )

    if not isinstance(points, np.ndarray):
        raise TypeError(f"points must be np.ndarray, got {type(points).__name__}")

    if points.ndim != 2:
        raise ValueError(
            f"points must be 2D array with shape (n, dims), "
            f"got {points.ndim}D array with shape {points.shape}"
        )

    if points.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"points dtype must be float32 or float64, got {points.dtype}. "
            f"Convert with points.astype(np.float32) or points.astype(np.float64)"
        )

    dims = points.shape[1]
    if dims != 3:
        raise ValueError(f"points must have 3 dimensions, got {dims}")

    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    if not isinstance(vertex_link, OffsetBlockedArray):
        raise TypeError(
            f"vertex_link must be OffsetBlockedArray, "
            f"got {type(vertex_link).__name__}"
        )

    suffix = topology_suffix(
        vertex_link.offsets.dtype,
        real_dtype=points.dtype,
        dims=dims
    )
    func_name = f"taubin_smoothed_{suffix}"
    cpp_func = getattr(_trueform.geometry, func_name)

    return cpp_func(points, vertex_link._wrapper, iterations, lambda_, kpb)
