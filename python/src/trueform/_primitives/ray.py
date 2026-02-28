"""
Ray primitive

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .primitive import Primitive, PrimitiveType


class Ray(Primitive):
    """
    A ray defined by an origin and direction.

    The direction vector is stored as-is, NOT normalized.

    Parameters
    ----------
    data : array-like, optional
        Ray data. Shape (2, D) for a single ray, (N, 2, D) for a batch,
        where D is 2 or 3. Row 0 is origin, row 1 is direction.
    origin : array-like, optional
        Origin point(s). Shape (D,) for single, (N, D) for batch.
    direction : array-like, optional
        Direction vector(s). Shape (D,) for single, (N, D) for batch.

    Either ``data`` or both ``origin`` and ``direction`` must be provided.

    Examples
    --------
    >>> from trueform import Ray
    >>> ray = Ray(origin=[0, 0, 0], direction=[1, 2, 3])
    >>> ray.origin
    array([0., 0., 0.], dtype=float32)
    >>> ray.direction
    array([1., 2., 3.], dtype=float32)
    >>> ray = Ray([[0, 0, 0], [1, 1, 0]])
    >>> ray.origin
    array([0., 0., 0.], dtype=float32)

    Batch construction:

    >>> import numpy as np
    >>> rays = Ray(np.random.rand(50, 2, 3))
    >>> rays.count
    50
    """

    def __init__(self, data=None, *, origin=None, direction=None):
        if data is None:
            origin = np.asarray(origin)
            direction = np.asarray(direction)
            data = np.stack([origin, direction], axis=-2)
        else:
            data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        dims = data.shape[-1]
        if dims not in (2, 3):
            raise ValueError(f"Ray must be 2D or 3D, got {dims} dimensions")
        self._init_wrapper(data, PrimitiveType.ray, dims)

    @property
    def origin(self) -> np.ndarray:
        """Get ray origin(s)."""
        if self.is_batch:
            return self._data[:, 0]
        return self._data[0]

    @property
    def direction(self) -> np.ndarray:
        """Get ray direction(s)."""
        if self.is_batch:
            return self._data[:, 1]
        return self._data[1]

    @property
    def normalized_direction(self) -> np.ndarray:
        """Get unit direction vector(s)."""
        d = self.direction
        if self.is_batch:
            return d / np.linalg.norm(d, axis=-1, keepdims=True)
        return d / np.linalg.norm(d)

    @classmethod
    def from_points(cls, start, through):
        """
        Create ray from start point through another point.

        Parameters
        ----------
        start : array-like
            Ray origin, shape (D,)
        through : array-like
            Point the ray passes through, shape (D,)

        Returns
        -------
        Ray

        Examples
        --------
        >>> ray = Ray.from_points([0, 0], [1, 1])
        >>> ray.direction
        array([1., 1.], dtype=float32)
        """
        start = np.asarray(start)
        through = np.asarray(through)
        return cls(origin=start, direction=through - start)

    def __repr__(self) -> str:
        if self.is_batch:
            return f"Ray(batch={self.count}, dims={self._dims}, dtype={self._dtype})"
        return f"Ray(origin={self.origin.tolist()}, direction={self.direction.tolist()}, dtype={self._dtype})"
