"""
Face normals computation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import ensure_mesh


def normals(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]
) -> np.ndarray:
    """
    Compute face normals for a mesh.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: Mesh object
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    normals : np.ndarray of shape (num_faces, 3)
        Unit face normals

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
    >>> n = tf.normals(mesh)
    >>>
    >>> # From tuple
    >>> n = tf.normals((faces, points))
    """
    mesh = ensure_mesh(data, dims=3)
    return mesh.normals
