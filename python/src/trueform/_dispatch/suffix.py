"""
Suffix builders for C++ function name construction.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
import numpy as np


_DTYPE_MAP = {
    np.dtype('int32'): 'int',
    np.dtype('int64'): 'int64',
    np.dtype('float32'): 'float',
    np.dtype('float64'): 'double',
}


def dtype_str(dtype: np.dtype) -> str:
    """Map numpy dtype to C++ suffix component."""
    return _DTYPE_MAP[dtype]


def build_suffix(meta) -> str:
    """
    Build suffix for single-form operations.

    Pattern: {index?}{ngon?}{real}{dims}d
    Components omitted when None.

    Examples:
        PointCloud  -> float3d
        EdgeMesh    -> intfloat3d
        Mesh        -> int3float3d
    """
    parts = []
    if meta.index_dtype is not None:
        parts.append(dtype_str(meta.index_dtype))
    if meta.ngon is not None:
        parts.append(meta.ngon)
    parts.append(dtype_str(meta.real_dtype))
    parts.append(f"{meta.dims}d")
    return "".join(parts)


def build_suffix_pair(meta0, meta1) -> str:
    """
    Build suffix for form×form operations.

    Pattern: {idx0?}{idx1?}{ngon0?}{ngon1?}{real}{dims}d
    Components omitted when None.

    Examples:
        PointCloud × PointCloud -> float3d
        EdgeMesh × PointCloud   -> intfloat3d
        EdgeMesh × EdgeMesh     -> intintfloat3d
        Mesh × PointCloud       -> int3float3d
        Mesh × EdgeMesh         -> intint3float3d
        Mesh × Mesh             -> intint33float3d
    """
    parts = []
    if meta0.index_dtype is not None:
        parts.append(dtype_str(meta0.index_dtype))
    if meta1.index_dtype is not None:
        parts.append(dtype_str(meta1.index_dtype))
    if meta0.ngon is not None:
        parts.append(meta0.ngon)
    if meta1.ngon is not None:
        parts.append(meta1.ngon)
    parts.append(dtype_str(meta0.real_dtype))  # shared
    parts.append(f"{meta0.dims}d")             # shared
    return "".join(parts)


def topology_suffix(
    index_dtype: np.dtype,
    ngon: str = None,
    *,
    real_dtype: np.dtype = None,
    dims: int = None
) -> str:
    """
    Suffix for topology operations with dtype-based components.

    Pattern: {index}[_{ngon}][_{real}][_{dims}]
    Components are joined with underscores, omitted when None.

    Examples:
        topology_suffix(int32)                              -> 'int'
        topology_suffix(int32, '3')                         -> 'int_3'
        topology_suffix(int32, real_dtype=float32, dims=3)  -> 'int_float_3'

    Used by: k_ring, neighborhoods, boundary_edges, vertex_link, face_link, etc.
    """
    parts = [dtype_str(index_dtype)]
    if ngon is not None:
        parts.append(ngon)
    if real_dtype is not None:
        parts.append(dtype_str(real_dtype))
    if dims is not None:
        parts.append(str(dims))
    return "_".join(parts)


def connectivity_suffix(storage: str, index_dtype: np.dtype) -> str:
    """
    Suffix for connectivity/graph operations.
    Pattern: {storage}_{index}
    Used by: label_connected_components
    """
    return f"{storage}_{dtype_str(index_dtype)}"
