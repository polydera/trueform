"""
Line primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Optional


class Line:
    """
    A line defined by origin and direction.

    The direction vector is stored as-is without normalization.

    Parameters
    ----------
    origin : np.ndarray, optional
        Line origin, shape (D,) where D is 2 or 3
    direction : np.ndarray, optional
        Line direction, shape (D,)
    data : np.ndarray, optional
        Alternative: shape (2, D) with [origin, direction]

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import Line
    >>> # Using origin/direction
    >>> line = Line(origin=[0, 0, 0], direction=[1, 2, 3])
    >>> line.origin
    array([0., 0., 0.], dtype=float32)
    >>> line.direction
    array([1., 2., 3.], dtype=float32)
    >>> # Using data array
    >>> line = Line(data=[[0, 0, 0], [1, 1, 0]])
    """

    def __init__(self,
                 origin: Optional[np.ndarray] = None,
                 direction: Optional[np.ndarray] = None,
                 data: Optional[np.ndarray] = None):

        if data is not None:
            # Use data array directly
            data = np.asarray(data)
            if data.shape[0] != 2:
                raise ValueError(f"Line data must have shape (2, D), got {data.shape}")
            line_data = data.copy()
        elif origin is not None and direction is not None:
            # Construct from origin/direction
            origin_arr = np.asarray(origin)
            direction_arr = np.asarray(direction)

            if origin_arr.shape != direction_arr.shape:
                raise ValueError(f"Origin and direction must have same shape, got {origin_arr.shape} and {direction_arr.shape}")

            if origin_arr.ndim != 1:
                raise ValueError(f"Origin and direction must be 1D arrays, got shape {origin_arr.shape}")

            line_data = np.stack([origin_arr, direction_arr], axis=0)
        else:
            raise ValueError("Must provide either 'data' or both 'origin' and 'direction'")

        # Validate shape
        if line_data.ndim != 2 or line_data.shape[0] != 2:
            raise ValueError(f"Line data must have shape (2, D), got {line_data.shape}")

        # Validate dimensionality
        dims = line_data.shape[1]
        if dims not in [2, 3]:
            raise ValueError(f"Line must be 2D or 3D, got {dims} dimensions")

        # Validate dtype
        if line_data.dtype not in [np.float32, np.float64]:
            # Try to convert
            line_data = line_data.astype(np.float32)

        # Validate direction is not zero
        direction_norm = np.linalg.norm(line_data[1])
        if direction_norm < 1e-10:
            raise ValueError("Line direction vector cannot be zero")

        # Ensure C-contiguous
        if not line_data.flags['C_CONTIGUOUS']:
            line_data = np.ascontiguousarray(line_data)

        self._data = line_data
        self._dims = dims
        self._dtype = line_data.dtype

    @property
    def origin(self) -> np.ndarray:
        """Get line origin."""
        return self._data[0]

    @property
    def direction(self) -> np.ndarray:
        """Get line direction."""
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

    def __repr__(self) -> str:
        return f"Line(origin={self.origin.tolist()}, direction={self.direction.tolist()}, dtype={self._dtype})"
