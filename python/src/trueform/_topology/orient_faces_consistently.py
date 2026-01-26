"""
orient_faces_consistently() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix


def orient_faces_consistently(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh]
) -> Union[np.ndarray, OffsetBlockedArray]:
    """
    Orient all faces consistently using face areas as weights.

    Makes all faces in each connected manifold region have consistent winding
    order. The orientation is determined by the total area of faces - the
    majority orientation (by area) is preserved.

    Parameters
    ----------
    data : tuple or Mesh
        Input geometric data:
        - Tuple (faces, points) where:
          * faces: shape (N, 3) with dtype int32 or int64, or OffsetBlockedArray for dynamic
          * points: shape (M, Dims) where Dims = 2 or 3
        - Mesh: tf.Mesh object (2D or 3D, triangular or dynamic)

    Returns
    -------
    np.ndarray or OffsetBlockedArray
        Reoriented faces array with same shape and dtype as input.
        Returns OffsetBlockedArray for dynamic meshes.
        Points are not returned as they remain unchanged.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create mesh with inconsistent face orientations
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)  # second face flipped
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]], dtype=np.float32)
    >>>
    >>> # Orient faces consistently
    >>> new_faces = tf.orient_faces_consistently((faces, points))
    >>> # Faces now have consistent winding order
    >>>
    >>> # With Mesh object
    >>> mesh = tf.Mesh(faces, points)
    >>> new_faces = tf.orient_faces_consistently(mesh)

    Notes
    -----
    - Only manifold edges (shared by exactly 2 faces) are used to propagate
      orientation. Non-manifold edges act as barriers between regions.
    - Each connected region is oriented independently based on its face areas.
    - The function uses squared areas for efficiency (avoids sqrt).
    """

    # Handle input
    if isinstance(data, Mesh):
        mesh = data
        has_mel = data._wrapper.has_manifold_edge_link()
    elif isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple must have exactly 2 elements (faces, points), got {len(data)}"
            )
        mesh = Mesh(data[0], data[1])  # Mesh validates everything
        has_mel = False
    else:
        raise TypeError(
            f"Expected tuple (faces, points) or Mesh, got {type(data).__name__}"
        )

    # Validate ngon - only triangles (3) or dynamic allowed
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"mesh must have triangular faces or be dynamic, got {mesh.ngon} vertices per face. "
            f"orient_faces_consistently only supports triangles and dynamic meshes."
        )

    # Create new mesh with copied faces
    if mesh.is_dynamic:
        faces_copy = OffsetBlockedArray(
            mesh.faces.offsets.copy(),
            mesh.faces.data.copy()
        )
    else:
        faces_copy = mesh.faces.copy()
    new_mesh = Mesh(faces_copy, mesh.points)

    # Copy manifold_edge_link if available
    if has_mel:
        new_mesh.manifold_edge_link = mesh.manifold_edge_link

    # Build dispatch suffix
    ngon = 'dyn' if new_mesh.is_dynamic else '3'
    meta = InputMeta(new_mesh.faces.dtype, new_mesh.points.dtype, ngon, new_mesh.dims)
    suffix = build_suffix(meta)

    # Call C++
    func_name = f"orient_faces_consistently_{suffix}"
    getattr(_trueform.topology, func_name)(new_mesh._wrapper)

    return new_mesh.faces
