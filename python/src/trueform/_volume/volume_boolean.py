"""
Boolean CSG of signed distance fields

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np

from .. import _trueform
from .volume import Volume, _DTYPE_CODES, _ROWS, _resolve_dtype


_OP_CODES = {
    "union": 0,
    "intersection": 1,
    "difference": 2,
}


def volume_boolean(a: Volume, b: Volume, operation: str = "union",
                   dtype=None) -> Volume:
    """
    Boolean CSG of two signed distance fields, returned as a new field.

    Combines two SDFs voxel-wise under the negative-inside convention:
    ``union = min(a, b)``, ``intersection = max(a, b)``,
    ``difference = max(a, -b)``. The zero level set of the result is exactly
    the boolean of the two input solids — extract it with :func:`isosurface`.

    The combine happens on one shared grid. Matching grids **and** matching
    :attr:`Volume.transformation` poses combine voxel-wise (exact) and the
    result keeps that shared pose; anything else — a different grid, a
    different pose — resamples both operands onto a common world-axis-aligned
    grid spanning the union of their world domains at the finer of the two
    local spacings, and the result comes back unposed. Out of an operand's
    domain is outside its solid: each operand reads a far-outside sentinel
    past its own box, so a solid that reaches its own domain boundary stops
    there rather than continuing across the shared grid.

    Parameters
    ----------
    a : Volume
        The first field (A).
    b : Volume
        The second field (B). Must have the same dtype as `a`.
    operation : str, optional
        ``"union"`` (default), ``"intersection"``, or ``"difference"``
        (A minus B).
    dtype : numpy.dtype, optional
        The sample type the combined field is computed and stored in, float32
        or float64. Default: A's own ``coordinate_dtype``.

    Returns
    -------
    Volume
        A new volume holding the combined field, always real-valued in
        `dtype`: the combine decides in a floating type, so an int16-sampled
        pair returns the requested float field rather than int16 samples. Its
        :attr:`Volume.transformation` is the operands' shared pose where they
        shared one, and None otherwise — the general result stands on the
        world axes.

    Raises
    ------
    TypeError
        If the operands' sample dtypes differ, or if they are unsigned. An SDF
        has a negative inside, so an unsigned field is not one and no
        operation combines it.

    Examples
    --------
    >>> import trueform as tf
    >>> a = tf.sphere_sdf((32, 32, 32), (0.5,) * 3, (0,) * 3, (6.0, 8.0, 8.0), 4.0)
    >>> b = tf.sphere_sdf((32, 32, 32), (0.5,) * 3, (0,) * 3, (10.0, 8.0, 8.0), 4.0)
    >>> merged = tf.volume_boolean(a, b, "union")
    >>> faces, points = tf.isosurface(merged)
    """
    if not isinstance(a, Volume) or not isinstance(b, Volume):
        raise TypeError(
            f"Expected two Volumes, got {type(a).__name__} and {type(b).__name__}"
        )
    if a.dtype != b.dtype:
        raise TypeError(
            f"Volume dtypes must match, got {a.dtype} and {b.dtype}"
        )

    row = _ROWS[a.dtype]
    if row.boolean is None:
        accepted = ", ".join(
            str(d) for d, r in _ROWS.items() if r.boolean is not None
        )
        raise TypeError(
            f"volume_boolean: an SDF has a negative inside, so a {a.dtype} "
            f"field is not one; accepted sample dtypes are {accepted}. "
            f"Convert the mask to a signed field first."
        )

    code = _OP_CODES.get(operation)
    if code is None:
        raise ValueError(
            f"volume_boolean: unknown operation {operation!r}; "
            f"expected 'union', 'intersection', or 'difference'"
        )

    out_dtype = _resolve_dtype(dtype, a.coordinate_dtype)
    func = getattr(_trueform.volume, row.boolean)
    flat, dims, spacing, origin, pose = func(a._wrapper, b._wrapper, code,
                                             _DTYPE_CODES[out_dtype])
    out = Volume._from_parts(flat, dims, spacing, origin)
    if pose is not None:
        # The setter reads an identity as unposed, which is what the
        # world-axis-aligned general result states.
        out.transformation = np.asarray(pose).astype(out.coordinate_dtype)
    return out
