"""
Error-budget mesh simplification

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


def simplified(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray]],
    *,
    error_rel: float = 0.002,
    optimize_iterations: int = 3,
    iterations: int = 1,
    relaxation_iters: int = 3,
    lambda_: float = 0.5,
    min_quality: float = 0.3,
    preserve_boundary: bool = True,
    stabilizer: float = 1e-3,
    parallel: bool = True,
    feature_angle: float = -1.0,
    feature_weight: float = 100.0,
    preserve_regions: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simplify a triangle mesh to an error budget using quadric error metrics.

    Every edge whose quadric error is within ``error_rel`` times the mesh
    bounding-box diagonal is collapsed, cheapest first, to exhaustion. Unlike
    decimation there is no target face count: flat regions collapse for almost
    no error while curved detail and feature edges survive.

    Parameters
    ----------
    data : Mesh or (faces, points) tuple
        Triangle mesh (ngon=3, 3D).
    error_rel : float, optional
        Error budget as a fraction of the mesh bounding-box diagonal. An edge
        is collapsed when its quadric error is <= error_rel * diagonal.
        Default is 0.002.
    optimize_iterations : int, optional
        Quality cleanup after each collapse: this many rounds of min-angle edge
        flip and tangential relaxation (feature/region/boundary-aware). 0 is a
        pure error-budget collapse with no flips. Default is 3.
    iterations : int, optional
        Outer collapse rounds. 1 is a single collapse + cleanup (plain
        simplify). >1 re-collapses after each cleanup -- the flip pass unblocks
        collapses the quality guard had stopped, removing more (an iterated
        remesh), at the cost of more deviation from the original. Default is 1.
    relaxation_iters : int, optional
        Tangential relaxation passes per cleanup round. Default is 3.
    lambda_ : float, optional
        Damping factor for tangential relaxation in (0, 1]. Default is 0.5.
    min_quality : float, optional
        Worst triangle quality allowed after a collapse, in [0, 1] (1 =
        equilateral, ->0 = sliver). Negative disables the check, 0 never worsens
        the worst surrounding triangle, >0 is a quality floor. Default is 0.3.
    preserve_boundary : bool, optional
        If True, boundary edges are never collapsed. Default is True.
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
    preserve_regions : np.ndarray, optional
        Per-face region labels (one int per input face). Edges between
        differing labels are preserved as features, and the function returns
        the per-face labels of the output mesh as a third value. Default is
        None.

    Returns
    -------
    faces : np.ndarray of shape (N, 3)
        Face indices of the simplified mesh.
    points : np.ndarray of shape (M, dims)
        Vertex positions of the simplified mesh.
    labels : np.ndarray of shape (N,), int32
        Per-face region labels of the output mesh. Returned only when
        ``preserve_regions`` is given.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(*tf.read_stl("model.stl"))
    >>> faces, points = tf.simplified(mesh)
    >>> faces, points = tf.simplified(mesh, error_rel=0.01, feature_angle=30)
    >>> faces, points, labels = tf.simplified(mesh, preserve_regions=region_labels)
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
            f"Simplification requires triangle meshes (ngon=3), got ngon={data.ngon}"
        )

    if data.dims != 3:
        raise ValueError(
            f"Simplification requires 3D meshes, got {data.dims}D"
        )

    meta = extract_meta(data)
    suffix = build_suffix(meta)
    func_name = f"simplified_{suffix}"
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
        error_rel,
        optimize_iterations,
        min_quality,
        preserve_boundary,
        stabilizer,
        parallel,
        fa,
        feature_weight,
        iterations,
        relaxation_iters,
        lambda_,
        regions
    )
