"""
Quadric error decimation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import math
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._dispatch import extract_meta, build_suffix


def decimated(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray]],
    target_proportion: float,
    *,
    max_aspect_ratio: float = 40.0,
    preserve_boundary: bool = False,
    stabilizer: float = 1e-3,
    parallel: bool = True,
    feature_angle: float = -1.0,
    feature_weight: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate a triangle mesh using quadric error metrics.

    Reduces the face count to approximately target_proportion of the original.

    Parameters
    ----------
    data : Mesh or (faces, points) tuple
        Triangle mesh (ngon=3, 3D).
    target_proportion : float
        Target face count as a fraction of the original (0.0 to 1.0).
    max_aspect_ratio : float, optional
        Maximum aspect ratio allowed after a collapse. Set negative to disable.
        Default is 40.0.
    preserve_boundary : bool, optional
        If True, boundary edges are never collapsed. Default is False.
    stabilizer : float, optional
        Tikhonov stabilizer for quadric solve. Default is 1e-3.
    parallel : bool, optional
        If True, use parallel partitioned collapse. Default is True.
    feature_angle : float, optional
        Feature edge detection angle in degrees. Edges sharper than this are
        preserved. Set negative to disable. Default is -1.0.
    feature_weight : float, optional
        Penalty weight for feature edge quadrics. Higher values preserve
        features more rigidly. Default is 100.0.

    Returns
    -------
    faces : np.ndarray of shape (N, 3)
        Face indices of the decimated mesh.
    points : np.ndarray of shape (M, dims)
        Vertex positions of the decimated mesh.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(*tf.read_stl("model.stl"))
    >>> faces, points = tf.decimated(mesh, 0.5)
    >>> faces, points = tf.decimated(mesh, 0.5, feature_angle=30)
    """

    if isinstance(data, tuple) and len(data) == 2:
        data = Mesh(*data)

    if not isinstance(data, Mesh):
        raise TypeError(
            f"data must be a Mesh or (faces, points) tuple, "
            f"got {type(data).__name__}"
        )

    if data.ngon != 3:
        raise ValueError(
            f"Decimation requires triangle meshes (ngon=3), got ngon={data.ngon}"
        )

    if data.dims != 3:
        raise ValueError(
            f"Decimation requires 3D meshes, got {data.dims}D"
        )

    meta = extract_meta(data)
    suffix = build_suffix(meta)
    func_name = f"decimated_{suffix}"
    cpp_func = getattr(_trueform.remesh, func_name)

    fa = math.radians(feature_angle) if feature_angle >= 0 else -1.0

    return cpp_func(
        data._wrapper,
        target_proportion,
        max_aspect_ratio,
        preserve_boundary,
        stabilizer,
        parallel,
        fa,
        feature_weight
    )
