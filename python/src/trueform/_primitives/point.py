"""
Point primitive

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .primitive import Primitive, PrimitiveType


class Point(Primitive):
    """
    A point in 2D or 3D space.

    Parameters
    ----------
    data : array-like
        Coordinates. Shape (D,) for a single point, (N, D) for a batch,
        where D is 2 or 3. Dtype float32 or float64 (other dtypes are
        cast to float32).

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import Point
    >>> p = Point([1.0, 2.0, 3.0])
    >>> p.dims
    3
    >>> p.coords
    array([1., 2., 3.], dtype=float32)

    Batch construction:

    >>> pts = Point(np.random.rand(100, 3))
    >>> pts.is_batch
    True
    >>> pts.count
    100
    """

    def __init__(self, data):
        data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        dims = data.shape[-1]
        if dims not in (2, 3):
            raise ValueError(f"Point must be 2D or 3D, got {dims} dimensions")
        self._init_wrapper(data, PrimitiveType.point, dims)

    @property
    def coords(self) -> np.ndarray:
        """Get point coordinates."""
        return self._data

    @property
    def x(self):
        """Get x coordinate. Returns scalar for single, array for batch."""
        if self.is_batch:
            return self._data[..., 0]
        return float(self._data[0])

    @property
    def y(self):
        """Get y coordinate. Returns scalar for single, array for batch."""
        if self.is_batch:
            return self._data[..., 1]
        return float(self._data[1])

    @property
    def z(self):
        """
        Get z coordinate. Returns scalar for single, array for batch.

        For 2D points, returns 0.0 (or array of zeros for batch).
        """
        if self._dims < 3:
            if self.is_batch:
                return np.zeros(self.count, dtype=self._dtype)
            return 0.0
        if self.is_batch:
            return self._data[..., 2]
        return float(self._data[2])

    def __repr__(self) -> str:
        if self.is_batch:
            return f"Point(batch={self.count}, dims={self._dims}, dtype={self._dtype})"
        return f"Point({self._data.tolist()}, dtype={self._dtype})"
