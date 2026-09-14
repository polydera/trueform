"""
Slice-plane isocontours of scalar volumes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple, Union
import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray
from .volume import Volume, _DTYPE_CODES, _ROWS, _resolve_dtype, _validate_triple


def volume_slice_contours(
    volume: Volume,
    plane_origin,
    u,
    v,
    dims2,
    spacing2,
    isovalues: Union[float, np.ndarray],
    dtype=None,
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Isocontours of a volume on an oriented slice plane, as 3D curves.

    Resamples the volume once onto the plane's 2D grid and contours every
    isovalue on that slice. The plane lives in the VOLUME'S LOCAL space:
    ``plane_origin`` is the position of slice grid node (0, 0) and ``u``, ``v``
    are the plane's unit axes; node (i, j) samples the volume at
    ``plane_origin + i * spacing2[0] * u + j * spacing2[1] * v``.

    Returns connected polylines, lifted back into the frame the volume states:
    world space when it carries a :attr:`Volume.transformation`, its own local
    frame when it does not — the plane's own origin and axes stay local either
    way. The crossings are welded, so the segments are joined topologically
    rather than by a coordinate search. Contours terminate cleanly at the
    volume boundary.

    Parameters
    ----------
    volume : Volume
        The scalar volume.
    plane_origin : sequence of 3 floats
        Local-space position of slice grid node (0, 0).
    u : sequence of 3 floats
        Unit 3D direction of the slice grid's first (i) axis.
    v : sequence of 3 floats
        Unit 3D direction of the slice grid's second (j) axis.
    dims2 : sequence of 2 ints
        Number of slice grid nodes along u and v.
    spacing2 : sequence of 2 floats
        Slice grid step along u and v.
    isovalues : float or array-like
        Single isovalue or array of isovalues, all contoured on the same slice.
        Field values in the type the call decides in, so an integer-sampled
        field takes a fractional threshold.
    dtype : numpy.dtype, optional
        The coordinate type the call decides and emits in, float32 or float64.
        Default: the volume's ``coordinate_dtype``.

    Returns
    -------
    paths : OffsetBlockedArray
        Connected polylines as indices into the points array; a closed contour
        repeats its first index last.
    points : np.ndarray of shape (P, 3)
        Contour point coordinates in the frame the volume states, in `dtype`.
        Empty if no isovalue is crossed.

    Examples
    --------
    >>> import trueform as tf
    >>> vol = tf.sphere_sdf((32, 32, 32), (0.5,) * 3, (0,) * 3, (8.0, 8.0, 8.0), 5.0)
    >>> paths, points = tf.volume_slice_contours(
    ...     vol, (0.0, 0.0, 8.0), (1, 0, 0), (0, 1, 0), (64, 64), (0.25, 0.25), 0.0)
    >>> len(paths) > 0
    True
    """
    if not isinstance(volume, Volume):
        raise TypeError(f"Expected Volume, got {type(volume).__name__}")

    plane_origin = _validate_triple(plane_origin, "plane_origin")
    u = _validate_triple(u, "u")
    v = _validate_triple(v, "v")

    dims2 = tuple(int(d) for d in dims2)
    if len(dims2) != 2:
        raise ValueError(f"dims2 must have length 2, got {len(dims2)}")
    spacing2 = tuple(float(s) for s in spacing2)
    if len(spacing2) != 2:
        raise ValueError(f"spacing2 must have length 2, got {len(spacing2)}")

    isovalue_array = np.ascontiguousarray(
        np.atleast_1d(np.asarray(isovalues, dtype=np.float64)).ravel()
    )

    out_dtype = _resolve_dtype(dtype, volume.coordinate_dtype)
    suffix = _ROWS[volume.dtype].suffix
    func = getattr(_trueform.volume, f"volume_slice_contours_{suffix}")
    (paths_offsets, paths_data), points = func(
        volume._wrapper, plane_origin, u, v, dims2, spacing2, isovalue_array,
        _DTYPE_CODES[out_dtype]
    )
    return OffsetBlockedArray(paths_offsets, paths_data), points
