"""
Triangle primitive

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .primitive import Primitive, PrimitiveType


class Triangle(Primitive):
    """
    A triangle defined by three vertices.

    Parameters
    ----------
    data : array-like, optional
        Vertex data. Shape (3, D) for a single triangle, (N, 3, D) for a
        batch, where D is 2 or 3.
    a : array-like, optional
        First vertex. Shape (D,) for single, (N, D) for batch.
    b : array-like, optional
        Second vertex. Shape (D,) for single, (N, D) for batch.
    c : array-like, optional
        Third vertex. Shape (D,) for single, (N, D) for batch.

    Either ``data`` or all of ``a``, ``b``, and ``c`` must be provided.

    Examples
    --------
    >>> from trueform import Triangle
    >>> tri = Triangle(a=[0, 0, 0], b=[1, 0, 0], c=[0, 1, 0])
    >>> tri.a
    array([0., 0., 0.], dtype=float32)

    Batch construction:

    >>> import numpy as np
    >>> tris = Triangle(np.random.rand(50, 3, 3))
    >>> tris.count
    50
    """

    def __init__(self, data=None, *, a=None, b=None, c=None):
        if data is None:
            data = np.stack([np.asarray(a), np.asarray(b), np.asarray(c)], axis=-2)
        else:
            data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        dims = data.shape[-1]
        if dims not in (2, 3):
            raise ValueError(f"Triangle must be 2D or 3D, got {dims} dimensions")
        if data.shape[-2] != 3:
            raise ValueError(
                f"Triangle must have 3 vertices (shape[-2] == 3), got {data.shape[-2]}")
        self._init_wrapper(data, PrimitiveType.triangle, dims)

    @property
    def a(self) -> np.ndarray:
        """Get first vertex."""
        if self.is_batch:
            return self._data[:, 0]
        return self._data[0]

    @property
    def b(self) -> np.ndarray:
        """Get second vertex."""
        if self.is_batch:
            return self._data[:, 1]
        return self._data[1]

    @property
    def c(self) -> np.ndarray:
        """Get third vertex."""
        if self.is_batch:
            return self._data[:, 2]
        return self._data[2]

    @property
    def vertices(self) -> np.ndarray:
        """Get all three vertices as (3, D) or (N, 3, D) array."""
        return self._data

    def __repr__(self) -> str:
        if self.is_batch:
            return f"Triangle(batch={self.count}, dims={self._dims}, dtype={self._dtype})"
        return f"Triangle(a={self.a.tolist()}, b={self.b.tolist()}, c={self.c.tolist()}, dtype={self._dtype})"
