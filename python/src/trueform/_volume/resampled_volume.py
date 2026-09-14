"""
Resample a scalar field onto a stated grid

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .. import _trueform
from .volume import Volume, _DTYPE_CODES, _ROWS, _resolve_dtype


def resampled_volume(volume: Volume, dims, spacing, origin,
                     dtype=None) -> Volume:
    """
    Resample a scalar field onto a stated grid, as a new field.

    Target node ``(i, j, k)`` sits at ``origin + index * spacing`` and takes
    the field's trilinear value there. An unposed volume regrids in its own
    local space with clamp-to-edge reads outside its domain; a volume carrying
    a :attr:`Volume.transformation` regrids **through** that pose — the target
    grid is world space, each node maps back through the pose into the field's
    local space, and a node outside the posed domain takes a sentinel above
    the field's maximum. This is the regrid behind :func:`volume_boolean`'s
    shared-grid path — use it directly to downsample a field for preview or to
    align two fields onto one grid.

    Parameters
    ----------
    volume : Volume
        The source field.
    dims : tuple of int
        Number of samples along x, y, z of the target grid.
    spacing : tuple of float
        Physical size of one target grid step along x, y, z.
    origin : tuple of float
        Position of target sample (0, 0, 0) — the volume's local space when it
        is unposed, world space when it is posed.
    dtype : numpy.dtype, optional
        The sample type the resampled field is computed and stored in,
        float32 or float64. Default: the volume's own sample dtype.

    Returns
    -------
    Volume
        A new volume holding the resampled field, unposed: it stands on the
        grid that was asked for, which for a posed source is a world-space
        one.

    Examples
    --------
    >>> import trueform as tf
    >>> field = tf.sphere_sdf((64, 64, 64), (0.25,) * 3, (0,) * 3,
    ...                       (8.0, 8.0, 8.0), 6.0)
    >>> coarse = tf.resampled_volume(field, (32, 32, 32), (0.5,) * 3,
    ...                              (0.0, 0.0, 0.0))
    """
    if not isinstance(volume, Volume):
        raise TypeError(f"Expected a Volume, got {type(volume).__name__}")

    row = _ROWS[volume.dtype]
    if row.resample is None:
        accepted = ", ".join(
            str(d) for d, r in _ROWS.items() if r.resample is not None
        )
        raise TypeError(
            f"resampled_volume: no regrid is registered for {volume.dtype} "
            f"samples; accepted sample dtypes are {accepted}."
        )

    out_dtype = _resolve_dtype(dtype, volume.dtype)
    func = getattr(_trueform.volume, row.resample)
    flat, out_dims, out_spacing, out_origin = func(
        volume._wrapper, tuple(int(d) for d in dims),
        tuple(float(s) for s in spacing), tuple(float(o) for o in origin),
        _DTYPE_CODES[out_dtype])
    return Volume._from_parts(flat, out_dims, out_spacing, out_origin)
