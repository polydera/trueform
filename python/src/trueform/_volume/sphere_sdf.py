"""
Sphere signed distance field generation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .. import _trueform
from .volume import Volume, _ROWS, _resolve_dtype, _validate_triple


def sphere_sdf(
    dims,
    spacing,
    origin,
    center,
    radius: float,
    *,
    dtype=np.float32,
) -> Volume:
    """
    Sample the signed distance field of a sphere onto a regular grid.

    Each sample holds ``distance(point, center) - radius``: negative inside
    the sphere, positive outside, zero on the surface. Extracting the
    isosurface at value 0 recovers the sphere.

    Parameters
    ----------
    dims : sequence of 3 ints
        Number of samples along x, y, z.
    spacing : sequence of 3 floats
        Physical size of one voxel step along x, y, z.
    origin : sequence of 3 floats
        Local-space position of sample (0, 0, 0).
    center : sequence of 3 floats
        Sphere center, in the volume's local frame.
    radius : float
        Sphere radius.
    dtype : numpy.dtype, optional
        Sample dtype, float32 (default) or float64.

    Returns
    -------
    Volume
        A volume holding the sphere SDF.

    Examples
    --------
    >>> import trueform as tf
    >>> vol = tf.sphere_sdf((32, 32, 32), (0.5, 0.5, 0.5), (0, 0, 0),
    ...                     (8.0, 8.0, 8.0), 5.0)
    >>> faces, points = tf.isosurface(vol)
    """
    dtype = _resolve_dtype(dtype, np.dtype(np.float32))

    dims = _validate_triple(dims, "dims", dtype=int)
    if any(d < 1 for d in dims):
        raise ValueError(f"dims must be at least 1 along each axis, got {dims}")
    spacing = _validate_triple(spacing, "spacing")
    origin = _validate_triple(origin, "origin")
    center = _validate_triple(center, "center")

    suffix = _ROWS[dtype].suffix
    func = getattr(_trueform.volume, f"make_sphere_sdf_{suffix}")
    flat, out_dims, out_spacing, out_origin = func(
        dims, spacing, origin, center, float(radius)
    )
    return Volume._from_parts(flat, out_dims, out_spacing, out_origin)
