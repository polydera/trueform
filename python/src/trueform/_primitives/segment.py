"""
Segment primitive

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .primitive import Primitive, PrimitiveType


class Segment(Primitive):
    """
    A line segment defined by two endpoints.

    Parameters
    ----------
    data : array-like, optional
        Endpoint data. Shape (2, D) for a single segment, (N, 2, D) for a
        batch, where D is 2 or 3.
    start : array-like, optional
        Start point(s). Shape (D,) for single, (N, D) for batch.
    end : array-like, optional
        End point(s). Shape (D,) for single, (N, D) for batch.

    Either ``data`` or both ``start`` and ``end`` must be provided.

    Examples
    --------
    >>> from trueform import Segment
    >>> seg = Segment(start=[0, 0, 0], end=[1, 1, 1])
    >>> seg.length
    1.7320508...
    >>> seg = Segment([[0, 0, 0], [1, 1, 1]])
    >>> seg.start
    array([0., 0., 0.], dtype=float32)

    Batch construction:

    >>> import numpy as np
    >>> segs = Segment(np.random.rand(50, 2, 3))
    >>> segs.count
    50
    """

    def __init__(self, data=None, *, start=None, end=None):
        if data is None:
            data = np.stack([np.asarray(start), np.asarray(end)], axis=-2)
        else:
            data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        dims = data.shape[-1]
        if dims not in (2, 3):
            raise ValueError(f"Segment must be 2D or 3D, got {dims} dimensions")
        self._init_wrapper(data, PrimitiveType.segment, dims)

    @property
    def start(self) -> np.ndarray:
        """Get start point(s)."""
        if self.is_batch:
            return self._data[:, 0]
        return self._data[0]

    @property
    def end(self) -> np.ndarray:
        """Get end point(s)."""
        if self.is_batch:
            return self._data[:, 1]
        return self._data[1]

    @property
    def endpoints(self) -> np.ndarray:
        """Get both endpoints as (2, D) or (N, 2, D) array."""
        return self._data

    @property
    def vector(self) -> np.ndarray:
        """Get direction vector from start to end."""
        return self.end - self.start

    @property
    def midpoint(self) -> np.ndarray:
        """Get midpoint of the segment."""
        return (self.start + self.end) / 2

    @property
    def length(self):
        """Get length of the segment. Returns scalar for single, array for batch."""
        v = self.vector
        if self.is_batch:
            return np.linalg.norm(v, axis=-1)
        return float(np.linalg.norm(v))

    def __repr__(self) -> str:
        if self.is_batch:
            return f"Segment(batch={self.count}, dims={self._dims}, dtype={self._dtype})"
        return f"Segment(start={self.start.tolist()}, end={self.end.tolist()}, dtype={self._dtype})"
