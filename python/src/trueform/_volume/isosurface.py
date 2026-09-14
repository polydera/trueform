"""
Isosurface extraction from scalar volumes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple
import numpy as np
from .. import _trueform
from .volume import Volume, _DTYPE_CODES, _ROWS, _resolve_dtype


_METHOD_CODES = {
    "flying_edges": 0,
    "dual_contouring": 1,
}


def isosurface(
    volume: Volume,
    iso: float = 0.0,
    *,
    method: str = "flying_edges",
    refine: bool = True,
    stabilizer: float = 0.01,
    dtype=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the isosurface of a scalar volume as a triangle mesh.

    Produces the level set ``field == iso`` as a welded, indexed triangle mesh
    in the frame the volume states: world space when it carries a
    :attr:`Volume.transformation`, its own local frame when it does not — and
    a reflecting pose keeps the winding outward. For a signed distance field,
    ``iso = 0`` recovers the surface and ``iso = +/-d`` gives the offset
    surface. A corner is treated as inside when ``sample < iso``, so an SDF
    (negative inside) yields outward-facing windings.

    Parameters
    ----------
    volume : Volume
        The scalar volume.
    iso : float, optional
        The isovalue to extract, default 0 (the SDF zero level set).
    method : str, optional
        ``"flying_edges"`` (default) places every vertex on a grid edge: the
        fastest regular output, defined for any scalar field.
        ``"dual_contouring"`` places one vertex per surface component of a
        cell, fitted to the field's own crossings, so creases and corners
        survive; it assumes a distance-like field.
    refine : bool, optional
        Dual contouring only: recover sharp features by refitting each feature
        vertex to the planes its neighbourhood's crossings state. Default True.
    stabilizer : float, optional
        Dual contouring only: the dimensionless weight of the pull toward the
        crossing centroid. Must be finite and nonnegative. Default 0.01.
    dtype : numpy.dtype, optional
        The coordinate type the call decides and emits in, float32 or float64.
        Default: the volume's ``coordinate_dtype``. An integer-sampled field
        emits float32 geometry — every sample is read through one cast into the
        deciding type, so the crossing lands where the field puts it.

    Returns
    -------
    faces : np.ndarray of shape (num_triangles, 3), dtype int32
        Triangle face indices.
    points : np.ndarray of shape (num_points, 3)
        Vertex coordinates in `dtype`. Empty if the isovalue is not crossed.

    Examples
    --------
    >>> import trueform as tf
    >>> vol = tf.sphere_sdf((32, 32, 32), (0.5, 0.5, 0.5), (0, 0, 0),
    ...                     (8.0, 8.0, 8.0), 5.0)
    >>> faces, points = tf.isosurface(vol)
    >>> sharp_faces, sharp_points = tf.isosurface(vol, method="dual_contouring")

    An int16 CT volume, thresholded at a Hounsfield value:

    >>> ct = tf.Volume(counts_int16, spacing=(0.7, 0.7, 1.5))  # doctest: +SKIP
    >>> faces, points = tf.isosurface(ct, 300.0)               # doctest: +SKIP
    """
    if not isinstance(volume, Volume):
        raise TypeError(f"Expected Volume, got {type(volume).__name__}")

    code = _METHOD_CODES.get(method)
    if code is None:
        raise ValueError(
            f"isosurface: unknown method {method!r}; "
            f"expected 'flying_edges' or 'dual_contouring'"
        )

    stabilizer = float(stabilizer)
    if not np.isfinite(stabilizer) or stabilizer < 0.0:
        raise ValueError(
            f"isosurface: stabilizer must be finite and nonnegative, "
            f"got {stabilizer}"
        )

    out_dtype = _resolve_dtype(dtype, volume.coordinate_dtype)
    suffix = _ROWS[volume.dtype].suffix
    func = getattr(_trueform.volume, f"isosurface_{suffix}")
    return func(volume._wrapper, float(iso), code, bool(refine), stabilizer,
                _DTYPE_CODES[out_dtype])
