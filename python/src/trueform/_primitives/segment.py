"""
Segment primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np


class Segment:
    """
    A line segment defined by two endpoints.

    Parameters
    ----------
    endpoints : np.ndarray
        Two endpoints, shape (2, D) where D is 2 or 3, dtype float32 or float64

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import Segment
    >>> seg = Segment([[0, 0, 0], [1, 1, 1]])
    >>> seg.start
    array([0., 0., 0.], dtype=float32)
    >>> seg.end
    array([1., 1., 1.], dtype=float32)
    """

    def __init__(self, endpoints: np.ndarray):
        # Convert to numpy if needed
        endpoints = np.asarray(endpoints)

        # Validate shape
        if endpoints.ndim != 2:
            raise ValueError(f"Segment endpoints must be 2D array, got shape {endpoints.shape}")

        if endpoints.shape[0] != 2:
            raise ValueError(f"Segment must have exactly 2 endpoints, got {endpoints.shape[0]}")

        # Validate dimensionality
        dims = endpoints.shape[1]
        if dims not in [2, 3]:
            raise ValueError(f"Segment must be 2D or 3D, got {dims} dimensions")

        # Validate dtype
        if endpoints.dtype not in [np.float32, np.float64]:
            # Try to convert
            endpoints = endpoints.astype(np.float32)

        # Ensure C-contiguous
        if not endpoints.flags['C_CONTIGUOUS']:
            endpoints = np.ascontiguousarray(endpoints)

        self._data = endpoints
        self._dims = dims
        self._dtype = endpoints.dtype

    @property
    def endpoints(self) -> np.ndarray:
        """Get both endpoints as (2, D) array."""
        return self._data

    @property
    def start(self) -> np.ndarray:
        """Get start point."""
        return self._data[0]

    @property
    def end(self) -> np.ndarray:
        """Get end point."""
        return self._data[1]

    @property
    def data(self) -> np.ndarray:
        """Get underlying data array."""
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
    def length(self) -> float:
        """Get length of the segment."""
        return float(np.linalg.norm(self._data[1] - self._data[0]))

    @property
    def vector(self) -> np.ndarray:
        """Get direction vector from start to end."""
        return self._data[1] - self._data[0]

    @property
    def midpoint(self) -> np.ndarray:
        """Get midpoint of the segment."""
        return (self._data[0] + self._data[1]) / 2

    @classmethod
    def from_points(cls, start, end):
        """
        Create segment from two points.

        Parameters
        ----------
        start : array-like
            Start point, shape (D,) where D is 2 or 3
        end : array-like
            End point, shape (D,) where D is 2 or 3

        Returns
        -------
        Segment
            Segment from start to end

        Examples
        --------
        >>> seg = Segment.from_points([0, 0], [1, 1])
        >>> seg.start
        array([0., 0.], dtype=float32)
        >>> seg.end
        array([1., 1.], dtype=float32)
        """
        start = np.asarray(start)
        end = np.asarray(end)
        return cls(np.stack([start, end], axis=0))

    def __repr__(self) -> str:
        return f"Segment(start={self.start.tolist()}, end={self.end.tolist()}, dtype={self._dtype})"
