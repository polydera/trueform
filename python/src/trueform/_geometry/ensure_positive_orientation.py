"""
ensure_positive_orientation() function implementation

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


def ensure_positive_orientation(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh],
    is_consistent: bool = False
) -> Union[np.ndarray, OffsetBlockedArray]:
    """
    Ensure mesh faces are oriented with outward-pointing normals.

    For closed 3D meshes, orients all faces consistently and ensures
    the signed volume is positive (normals point outward). This is a
    two-step process:
    1. Make faces locally consistent (unless is_consistent=True)
    2. Check signed volume and flip all faces if negative

    Parameters
    ----------
    data : tuple or Mesh
        Input geometric data:
        - Tuple (faces, points) where:
          * faces: shape (N, 3) with dtype int32 or int64, or OffsetBlockedArray for dynamic
          * points: shape (M, 3) - must be 3D
        - Mesh: tf.Mesh object (3D, triangular or dynamic)
    is_consistent : bool, optional
        If True, skip the orient_faces_consistently step. Use this when
        you know faces are already locally consistent and only need the
        global orientation check. Default is False.

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
    >>> # Create mesh with inverted faces (negative volume)
    >>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]], dtype=np.float32)
    >>>
    >>> # Ensure positive orientation
    >>> new_faces = tf.ensure_positive_orientation((faces, points))
    >>> # Faces now have outward-pointing normals
    >>>
    >>> # With Mesh object
    >>> mesh = tf.Mesh(faces, points)
    >>> new_faces = tf.ensure_positive_orientation(mesh)
    >>>
    >>> # Skip consistency step if already consistent
    >>> new_faces = tf.ensure_positive_orientation(mesh, is_consistent=True)

    Notes
    -----
    - This function only works with 3D meshes (uses signed volume calculation).
    - The signed volume is computed using the divergence theorem, which requires
      a closed mesh for meaningful results.
    - If the manifold_edge_link is already cached on the Mesh, it will be reused.
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

    # Validate 3D only
    if mesh.dims != 3:
        raise ValueError(
            f"ensure_positive_orientation requires 3D mesh, got {mesh.dims}D. "
            f"Signed volume calculation is only defined for 3D closed meshes."
        )

    # Validate ngon - only triangles (3) or dynamic allowed
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"mesh must have triangular faces or be dynamic, got {mesh.ngon} vertices per face. "
            f"ensure_positive_orientation only supports triangles and dynamic meshes."
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
    func_name = f"ensure_positive_orientation_{suffix}"
    getattr(_trueform.geometry, func_name)(new_mesh._wrapper, is_consistent)

    return new_mesh.faces
