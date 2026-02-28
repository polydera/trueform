"""
Polygon primitive

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .primitive import Primitive, PrimitiveType


class Polygon(Primitive):
    """
    A polygon defined by ordered vertices.

    Parameters
    ----------
    data : array-like
        Vertex data. Shape (V, D) for a single polygon, (N, V, D) for a
        batch, where V >= 3 and D is 2 or 3.

    Examples
    --------
    >>> from trueform import Polygon
    >>> quad = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> quad.num_vertices
    4
    >>> quad.dims
    2

    Batch construction:

    >>> import numpy as np
    >>> polys = Polygon(np.random.rand(50, 5, 3))
    >>> polys.count
    50
    """

    def __init__(self, data):
        data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        dims = data.shape[-1]
        if dims not in (2, 3):
            raise ValueError(f"Polygon must be 2D or 3D, got {dims} dimensions")
        self._init_wrapper(data, PrimitiveType.polygon, dims)

    @property
    def vertices(self) -> np.ndarray:
        """Get vertices as (V, D) or (N, V, D) array."""
        return self._data

    @property
    def num_vertices(self) -> int:
        """Get number of vertices per polygon."""
        return self._data.shape[-2]

    def __repr__(self) -> str:
        nv = self.num_vertices
        if self.is_batch:
            return f"Polygon(batch={self.count}, vertices={nv}, dims={self._dims}, dtype={self._dtype})"
        if nv <= 5:
            return f"Polygon({self._data.tolist()}, dtype={self._dtype})"
        return f"Polygon({nv} vertices, dims={self._dims}, dtype={self._dtype})"
