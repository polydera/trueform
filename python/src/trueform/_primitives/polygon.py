"""
Polygon primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np


class Polygon:
    """
    A polygon defined by ordered vertices.

    Parameters
    ----------
    vertices : np.ndarray
        Ordered vertices, shape (N, D) where N >= 3, D is 2 or 3, dtype float32 or float64

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import Polygon
    >>> tri = Polygon([[0, 0], [1, 0], [0.5, 1]])
    >>> tri.num_vertices
    3
    >>> tri.dims
    2
    """

    def __init__(self, vertices: np.ndarray):
        # Convert to numpy if needed
        vertices = np.asarray(vertices)

        # Validate shape
        if vertices.ndim != 2:
            raise ValueError(
                f"Polygon vertices must be 2D array, got shape {vertices.shape}")

        num_vertices = vertices.shape[0]
        if num_vertices < 3:
            raise ValueError(
                f"Polygon must have at least 3 vertices, got {num_vertices}")

        # Validate dimensionality
        dims = vertices.shape[1]
        if dims not in [2, 3]:
            raise ValueError(
                f"Polygon must be 2D or 3D, got {dims} dimensions")

        # Validate dtype
        if vertices.dtype not in [np.float32, np.float64]:
            # Try to convert
            vertices = vertices.astype(np.float32)

        # Ensure C-contiguous
        if not vertices.flags['C_CONTIGUOUS']:
            vertices = np.ascontiguousarray(vertices)

        self._data = vertices
        self._dims = dims
        self._dtype = vertices.dtype

    @property
    def vertices(self) -> np.ndarray:
        """Get all vertices as (N, D) array."""
        return self._data

    @property
    def data(self) -> np.ndarray:
        """Get underlying data array."""
        return self._data

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return self._data.shape[0]

    @property
    def dims(self) -> int:
        """Get dimensionality (2 or 3)."""
        return self._dims

    @property
    def dtype(self) -> np.dtype:
        """Get data type (float32 or float64)."""
        return self._dtype

    def __repr__(self) -> str:
        n_verts = self.num_vertices
        if n_verts <= 5:
            # Show actual vertices for small polygons
            return f"Polygon({self._data.tolist()}, dtype={self._dtype})"
        else:
            # Show summary for large polygons
            return f"Polygon({n_verts} vertices, {self._dims}D, dtype={self._dtype})"

    def __len__(self) -> int:
        """Get number of vertices (for len(polygon) syntax)."""
        return self.num_vertices
