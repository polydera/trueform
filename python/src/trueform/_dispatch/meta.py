"""
Unified metadata extraction for dispatch.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from typing import NamedTuple, Optional
import numpy as np


class InputMeta(NamedTuple):
    """Unified metadata for any geometric input."""
    index_dtype: Optional[np.dtype]  # None for PointCloud/points-only
    real_dtype: np.dtype             # np.float32 or np.float64
    ngon: Optional[str]              # None for PointCloud/EdgeMesh, '2'/'3'/'dyn' for Mesh/tuples
    dims: int                        # 2 or 3


def extract_meta(data) -> InputMeta:
    """Extract InputMeta from Form, tuple, or array."""
    from .._spatial import Mesh, EdgeMesh, PointCloud

    if isinstance(data, Mesh):
        return InputMeta(
            index_dtype=data.faces.dtype,
            real_dtype=data.dtype,
            ngon='dyn' if data.is_dynamic else str(data.ngon),
            dims=data.dims,
        )
    elif isinstance(data, EdgeMesh):
        return InputMeta(
            index_dtype=data.edges.dtype,
            real_dtype=data.dtype,
            ngon=None,  # EdgeMesh has no ngon
            dims=data.dims,
        )
    elif isinstance(data, PointCloud):
        return InputMeta(
            index_dtype=None,  # PointCloud has no index
            real_dtype=data.dtype,
            ngon=None,
            dims=data.dims,
        )
    elif isinstance(data, tuple):
        return _extract_meta_from_tuple(data)
    elif isinstance(data, np.ndarray):
        return _extract_meta_from_array(data)
    else:
        raise TypeError(f"Cannot extract meta from {type(data).__name__}")


def _extract_meta_from_tuple(data: tuple) -> InputMeta:
    """Extract meta from (indices, points) tuple."""
    from .._core import OffsetBlockedArray

    indices, points = data

    if isinstance(indices, OffsetBlockedArray):
        ngon = 'dyn'
        index_dtype = indices.dtype
    elif isinstance(indices, np.ndarray):
        V = indices.shape[1]
        ngon = str(V) if V in (2, 3) else str(V)
        index_dtype = indices.dtype
    else:
        raise TypeError("indices must be ndarray or OffsetBlockedArray")

    return InputMeta(
        index_dtype=index_dtype,
        real_dtype=points.dtype,
        ngon=ngon,
        dims=points.shape[1],
    )


def _extract_meta_from_array(data: np.ndarray) -> InputMeta:
    """Extract meta from points-only array."""
    return InputMeta(
        index_dtype=None,
        real_dtype=data.dtype,
        ngon=None,
        dims=data.shape[1],
    )
