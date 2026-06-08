"""
Isotropic remeshing

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple, Optional
import math
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._dispatch import extract_meta, build_suffix


def isotropic_remeshed(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray]],
    target_length: float,
    *,
    iterations: int = 3,
    relaxation_iters: int = 3,
    min_quality: float = 0.3,
    lambda_: float = 0.5,
    preserve_boundary: bool = True,
    use_quadric: bool = False,
    parallel: bool = True,
    feature_angle: float = -1.0,
    feature_weight: float = 100.0,
    preserve_regions: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Isotropic remeshing of a triangle mesh.

    Iteratively splits long edges, collapses short edges, flips edges to
    improve quality, and applies tangential relaxation.

    Parameters
    ----------
    data : Mesh or (faces, points) tuple
        Triangle mesh (ngon=3, 3D).
    target_length : float
        Target edge length. Edges longer than this are split, shorter are
        collapsed.
    iterations : int, optional
        Number of outer iterations (split + collapse + flip + relax).
        Default is 3.
    relaxation_iters : int, optional
        Number of tangential relaxation iterations per outer iteration.
        Default is 3.
    min_quality : float, optional
        Worst triangle quality allowed after a collapse, in [0, 1] (1 =
        equilateral). Negative disables, 0 never worsens the worst surrounding
        triangle, >0 is a quality floor. Default is 0.3.
    lambda_ : float, optional
        Damping factor for tangential relaxation in (0, 1]. Default is 0.5.
    preserve_boundary : bool, optional
        If True, boundary edges are never split or collapsed. Default is True.
    use_quadric : bool, optional
        If True, use quadric error metric for collapse vertex placement.
        Default is False.
    parallel : bool, optional
        If True, use parallel execution. Default is True.
    feature_angle : float, optional
        Feature edge detection angle in degrees. Edges sharper than this are
        preserved. Set negative to disable. Default is -1.0.
    feature_weight : float, optional
        Penalty weight for feature edge quadrics. Higher values preserve
        features more rigidly. Default is 100.0.
    preserve_regions : np.ndarray, optional
        Per-face region labels (one int per input face). Edges between
        differing labels are preserved as features, and the function returns
        the per-face labels of the output mesh as a third value. Default is
        None.

    Returns
    -------
    faces : np.ndarray of shape (N, 3)
        Face indices of the remeshed mesh.
    points : np.ndarray of shape (M, dims)
        Vertex positions of the remeshed mesh.
    labels : np.ndarray of shape (N,), int32
        Per-face region labels of the output mesh. Returned only when
        ``preserve_regions`` is given.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(*tf.read_stl("model.stl"))
    >>> faces, points = tf.isotropic_remeshed(mesh, 0.02)
    >>> faces, points = tf.isotropic_remeshed(mesh, 0.02, feature_angle=30)
    >>> faces, points, labels = tf.isotropic_remeshed(mesh, 0.02, preserve_regions=region_labels)
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
            f"Isotropic remeshing requires triangle meshes (ngon=3), "
            f"got ngon={data.ngon}"
        )

    if data.dims != 3:
        raise ValueError(
            f"Isotropic remeshing requires 3D meshes, got {data.dims}D"
        )

    meta = extract_meta(data)
    suffix = build_suffix(meta)
    func_name = f"isotropic_remeshed_{suffix}"
    cpp_func = getattr(_trueform.remesh, func_name)

    fa = math.radians(feature_angle) if feature_angle >= 0 else -1.0

    regions = None
    if preserve_regions is not None:
        regions = np.ascontiguousarray(preserve_regions, dtype=np.int32)
        if regions.ndim != 1 or regions.shape[0] != data.number_of_faces:
            raise ValueError(
                f"preserve_regions must be a 1-D array with one label per face "
                f"({data.number_of_faces}), got shape "
                f"{np.asarray(preserve_regions).shape}"
            )

    return cpp_func(
        data._wrapper,
        target_length,
        iterations,
        relaxation_iters,
        min_quality,
        lambda_,
        preserve_boundary,
        use_quadric,
        parallel,
        fa,
        feature_weight,
        regions
    )
