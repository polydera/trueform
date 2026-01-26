"""
Shape index computation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .principal_curvatures import principal_curvatures


def shape_index(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]],
    k: int = 2
) -> np.ndarray:
    """
    Compute shape index at each vertex.

    Shape index maps principal curvatures to a normalized scale [-1, 1]:
      - -1: spherical cup (concave)
      - -0.5: cylindrical cup
      -  0: saddle point
      -  0.5: cylindrical cap
      -  1: spherical cap (convex)

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: Mesh object
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh
    k : int, default 2
        k-ring neighborhood size for curvature estimation.

    Returns
    -------
    shape_index : np.ndarray of shape (num_points,)
        Shape index at each vertex, range [-1, 1].

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
    >>> si = tf.shape_index(mesh)
    >>>
    >>> # From tuple with custom k-ring
    >>> si = tf.shape_index((faces, points), k=3)
    """
    k0, k1 = principal_curvatures(data, k=k, directions=False)

    # S = (2/π) * arctan((k1 + k0) / (k1 - k0))
    return (2.0 / np.pi) * np.arctan2(k1 + k0, k1 - k0)
