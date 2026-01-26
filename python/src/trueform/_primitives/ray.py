"""
Ray primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Optional


class Ray:
    """
    A ray defined by origin and direction.

    ⚠️  **Important:** The direction vector is stored as-is, NOT normalized.
        Use the `normalized_direction` property if you need a unit vector.

    Parameters
    ----------
    origin : np.ndarray, optional
        Ray origin, shape (D,) where D is 2 or 3
    direction : np.ndarray, optional
        Ray direction, shape (D,)
    data : np.ndarray, optional
        Alternative: shape (2, D) with [origin, direction]

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import Ray
    >>> # Using origin/direction
    >>> ray = Ray(origin=[0, 0, 0], direction=[1, 2, 3])
    >>> ray.origin
    array([0., 0., 0.], dtype=float32)
    >>> ray.direction
    array([1., 2., 3.], dtype=float32)
    >>> # Using data array
    >>> ray = Ray(data=[[0, 0, 0], [1, 1, 0]])
    """

    def __init__(self,
                 origin: Optional[np.ndarray] = None,
                 direction: Optional[np.ndarray] = None,
                 data: Optional[np.ndarray] = None):

        if data is not None:
            # Use data array directly
            data = np.asarray(data)
            if data.shape[0] != 2:
                raise ValueError(f"Ray data must have shape (2, D), got {data.shape}")
            ray_data = data.copy()
        elif origin is not None and direction is not None:
            # Construct from origin/direction
            origin_arr = np.asarray(origin)
            direction_arr = np.asarray(direction)

            if origin_arr.shape != direction_arr.shape:
                raise ValueError(f"Origin and direction must have same shape, got {origin_arr.shape} and {direction_arr.shape}")

            if origin_arr.ndim != 1:
                raise ValueError(f"Origin and direction must be 1D arrays, got shape {origin_arr.shape}")

            ray_data = np.stack([origin_arr, direction_arr], axis=0)
        else:
            raise ValueError("Must provide either 'data' or both 'origin' and 'direction'")

        # Validate shape
        if ray_data.ndim != 2 or ray_data.shape[0] != 2:
            raise ValueError(f"Ray data must have shape (2, D), got {ray_data.shape}")

        # Validate dimensionality
        dims = ray_data.shape[1]
        if dims not in [2, 3]:
            raise ValueError(f"Ray must be 2D or 3D, got {dims} dimensions")

        # Validate dtype
        if ray_data.dtype not in [np.float32, np.float64]:
            # Try to convert
            ray_data = ray_data.astype(np.float32)

        # Validate direction is not zero
        direction_norm = np.linalg.norm(ray_data[1])
        if direction_norm < 1e-10:
            raise ValueError("Ray direction vector cannot be zero")

        # Ensure C-contiguous
        if not ray_data.flags['C_CONTIGUOUS']:
            ray_data = np.ascontiguousarray(ray_data)

        self._data = ray_data
        self._dims = dims
        self._dtype = ray_data.dtype

    @property
    def origin(self) -> np.ndarray:
        """Get ray origin."""
        return self._data[0]

    @property
    def direction(self) -> np.ndarray:
        """Get ray direction."""
        return self._data[1]

    @property
    def data(self) -> np.ndarray:
        """Get underlying data array as (2, D) with [origin, direction]."""
        return self._data

    @property
    def dims(self) -> int:
        """Get dimensionality (2 or 3)."""
        return self._dims

    @property
    def dtype(self) -> np.dtype:
        """Get data type (float32 or float64)."""
        return self._dtype

    @property
    def normalized_direction(self) -> np.ndarray:
        """Get unit direction vector (normalized to length 1)."""
        return self._data[1] / np.linalg.norm(self._data[1])

    @property
    def direction_norm(self) -> float:
        """Get magnitude (length) of the direction vector."""
        return float(np.linalg.norm(self._data[1]))

    @classmethod
    def from_points(cls, start, through_point):
        """
        Create ray from start point through another point.

        The direction is computed as (through_point - start).

        Parameters
        ----------
        start : array-like
            Ray origin point, shape (D,) where D is 2 or 3
        through_point : array-like
            Point the ray passes through, shape (D,)

        Returns
        -------
        Ray
            Ray starting at start and passing through through_point

        Examples
        --------
        >>> ray = Ray.from_points([0, 0], [1, 1])
        >>> ray.origin
        array([0., 0.], dtype=float32)
        >>> ray.direction
        array([1., 1.], dtype=float32)
        """
        start = np.asarray(start)
        through_point = np.asarray(through_point)
        direction = through_point - start
        return cls(origin=start, direction=direction)

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin.tolist()}, direction={self.direction.tolist()}, dtype={self._dtype})"
