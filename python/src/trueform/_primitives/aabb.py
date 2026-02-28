"""
AABB (Axis-Aligned Bounding Box) primitive

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .primitive import Primitive, PrimitiveType


class AABB(Primitive):
    """
    An axis-aligned bounding box defined by min and max corners.

    Parameters
    ----------
    data : array-like, optional
        Bounds data. Shape (2, D) for a single AABB, (N, 2, D) for a batch,
        where D is 2 or 3. Row 0 is min, row 1 is max.
    min : array-like, optional
        Minimum corner(s). Shape (D,) for single, (N, D) for batch.
    max : array-like, optional
        Maximum corner(s). Shape (D,) for single, (N, D) for batch.

    Either ``data`` or both ``min`` and ``max`` must be provided.

    Examples
    --------
    >>> from trueform import AABB
    >>> box = AABB(min=[0, 0, 0], max=[1, 1, 1])
    >>> box.center
    array([0.5, 0.5, 0.5], dtype=float32)
    >>> box = AABB([[0, 0, 0], [1, 1, 1]])
    >>> box.min
    array([0., 0., 0.], dtype=float32)

    Batch construction:

    >>> import numpy as np
    >>> boxes = AABB(np.random.rand(50, 2, 3))
    >>> boxes.count
    50
    """

    def __init__(self, data=None, *, min=None, max=None):
        if data is None:
            data = np.stack([np.asarray(min), np.asarray(max)], axis=-2)
        else:
            data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        dims = data.shape[-1]
        if dims not in (2, 3):
            raise ValueError(f"AABB must be 2D or 3D, got {dims} dimensions")
        self._init_wrapper(data, PrimitiveType.aabb, dims)

    @property
    def min(self) -> np.ndarray:
        """Get minimum corner(s)."""
        if self.is_batch:
            return self._data[:, 0]
        return self._data[0]

    @property
    def max(self) -> np.ndarray:
        """Get maximum corner(s)."""
        if self.is_batch:
            return self._data[:, 1]
        return self._data[1]

    @property
    def bounds(self) -> np.ndarray:
        """Get bounds as (2, D) or (N, 2, D) array."""
        return self._data

    @property
    def center(self) -> np.ndarray:
        """Get center point(s)."""
        return (self.min + self.max) / 2

    @property
    def size(self) -> np.ndarray:
        """Get size in each dimension."""
        return self.max - self.min

    @property
    def volume(self):
        """Get volume (area in 2D). Returns scalar for single, array for batch."""
        s = self.size
        if self.is_batch:
            return np.prod(s, axis=-1)
        return float(np.prod(s))

    @classmethod
    def from_center_size(cls, center, size):
        """
        Create AABB from center point and size.

        Parameters
        ----------
        center : array-like
            Center point, shape (D,)
        size : array-like
            Size in each dimension, shape (D,)

        Returns
        -------
        AABB

        Examples
        --------
        >>> box = AABB.from_center_size([5, 5], [2, 2])
        >>> box.min
        array([4., 4.], dtype=float32)
        """
        center = np.asarray(center)
        half = np.asarray(size) / 2
        return cls(min=center - half, max=center + half)

    @classmethod
    def from_points(cls, points):
        """
        Create AABB bounding the given points.

        Parameters
        ----------
        points : array-like
            Points, shape (N, D)

        Returns
        -------
        AABB

        Examples
        --------
        >>> box = AABB.from_points([[0, 0], [1, 0], [1, 1]])
        >>> box.max
        array([1., 1.], dtype=float32)
        """
        points = np.asarray(points)
        return cls(min=points.min(axis=0), max=points.max(axis=0))

    def __repr__(self) -> str:
        if self.is_batch:
            return f"AABB(batch={self.count}, dims={self._dims}, dtype={self._dtype})"
        return f"AABB(min={self.min.tolist()}, max={self.max.tolist()}, dtype={self._dtype})"
