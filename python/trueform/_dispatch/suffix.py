"""
Suffix builders for C++ function name construction.

Each function corresponds to a dispatch pattern used by one or more modules.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
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


def primitive_suffix(real_dtype: np.dtype, dims: int) -> str:
    """
    Suffix for primitive-only operations.
    Pattern: {real}{dims}d
    Used by: core.intersects, core.distance, core.ray_cast
    """
    return f"{dtype_str(real_dtype)}{dims}d"


def form_primitive_suffix(meta) -> str:
    """
    Suffix for form-primitive operations.
    Pattern: {index}{real}{ngon}{dims}d  (Mesh: ngon = '3' or 'dyn')
             {index}{real}{dims}d        (EdgeMesh: ngon not in suffix)
             {real}{dims}d               (PointCloud)
    Used by: spatial.neighbor_search, spatial.gather_ids, spatial.intersects
    """
    if meta.ngon is None:  # PointCloud
        return f"{dtype_str(meta.real_dtype)}{meta.dims}d"
    if meta.ngon == '2':  # EdgeMesh - doesn't include ngon in suffix
        return f"{dtype_str(meta.index_dtype)}{dtype_str(meta.real_dtype)}{meta.dims}d"
    # Mesh - includes ngon ('3' or 'dyn')
    return f"{dtype_str(meta.index_dtype)}{dtype_str(meta.real_dtype)}{meta.ngon}{meta.dims}d"


def form_form_suffix(meta0, meta1, form0_type, form1_type) -> str:
    """
    Suffix for form-form operations (varies by combination).
    Used by: spatial.gather_ids, spatial.intersects

    Note: Caller must handle symmetry (needs_swap) and index canonicalization
    before calling this function.
    """
    # Import locally to avoid cycles
    from .._spatial import Mesh, EdgeMesh, PointCloud

    real = dtype_str(meta0.real_dtype)
    dims = f"{meta0.dims}d"

    # PointCloud x PointCloud: {real}{dims}d
    if form0_type is PointCloud and form1_type is PointCloud:
        return f"{real}{dims}"

    # EdgeMesh x EdgeMesh: {idx0}{idx1}{real}{dims}d
    if form0_type is EdgeMesh and form1_type is EdgeMesh:
        return f"{dtype_str(meta0.index_dtype)}{dtype_str(meta1.index_dtype)}{real}{dims}"

    # EdgeMesh x PointCloud: {idx0}{real}{dims}d
    if form0_type is EdgeMesh and form1_type is PointCloud:
        return f"{dtype_str(meta0.index_dtype)}{real}{dims}"

    # Mesh x PointCloud: {idx0}{real}{ngon0}{dims}d
    if form0_type is Mesh and form1_type is PointCloud:
        return f"{dtype_str(meta0.index_dtype)}{real}{meta0.ngon}{dims}"

    # Mesh x EdgeMesh: {idx0}{idx1}{real}{ngon0}{dims}d
    if form0_type is Mesh and form1_type is EdgeMesh:
        return f"{dtype_str(meta0.index_dtype)}{dtype_str(meta1.index_dtype)}{real}{meta0.ngon}{dims}"

    # Mesh x Mesh: {idx0}{idx1}{ngon0}{ngon1}{real}{dims}d
    if form0_type is Mesh and form1_type is Mesh:
        return f"{dtype_str(meta0.index_dtype)}{dtype_str(meta1.index_dtype)}{meta0.ngon}{meta1.ngon}{real}{dims}"

    raise ValueError(f"Unknown form-form: {form0_type.__name__} x {form1_type.__name__}")


def indexed_geometry_suffix(V: str, index_dtype: np.dtype,
                            real_dtype: np.dtype, dims: int) -> str:
    """
    Suffix for indexed geometry operations.
    Pattern: {V}{index}{real}{dims}d  where V = "2", "3", "dyn"
    Used by: clean.cleaned, reindex.*, intersect.isocontours
    """
    return f"{V}{dtype_str(index_dtype)}{dtype_str(real_dtype)}{dims}d"


def points_suffix(real_dtype: np.dtype, dims: int) -> str:
    """
    Suffix for point-only operations.
    Pattern: {real}{dims}d
    Used by: clean.cleaned (points), geometry.chamfer_error
    """
    return f"{dtype_str(real_dtype)}{dims}d"


def reindex_points_suffix(index_dtype: np.dtype, real_dtype: np.dtype, dims: int) -> str:
    """
    Suffix for reindex point operations (includes index type from ids array).
    Pattern: {index}{real}{dims}d
    Used by: reindex.reindex_by_ids (points), reindex.reindex_by_mask (points)
    """
    return f"{dtype_str(index_dtype)}{dtype_str(real_dtype)}{dims}d"


def soup_suffix(V: int, real_dtype: np.dtype, dims: int) -> str:
    """
    Suffix for polygon/segment soup operations.
    Pattern: {V}{real}{dims}d  where V = 2 or 3
    Used by: clean.cleaned (soups)
    """
    return f"{V}{dtype_str(real_dtype)}{dims}d"


def topology_suffix(container: str, index_dtype: np.dtype) -> str:
    """
    Suffix for topology graph operations.
    Pattern: {container}_{index}
    Used by: topology.label_connected_components
    """
    return f"{container}_{dtype_str(index_dtype)}"


def topology_mesh_suffix(index_dtype: np.dtype, ngon: str) -> str:
    """
    Suffix for topology mesh operations.
    Pattern: {index}_{ngon}
    Used by: topology.boundary_edges, vertex_link, face_link, manifold_edge_link
    """
    return f"{dtype_str(index_dtype)}_{ngon}"


def isocontour_suffix(index_dtype: np.dtype, real_dtype: np.dtype, ngon: str, dims: int) -> str:
    """
    Suffix for isocontour and intersection curve operations.
    Pattern: {index}{ngon}{real}{dims}d
    Used by: intersect.isocontours, intersect.self_intersection_curves,
             cut.isobands, cut.embedded_self_intersection_curves,
             topology.orient_faces_consistently
    """
    return f"{dtype_str(index_dtype)}{ngon}{dtype_str(real_dtype)}{dims}d"


def boolean_suffix(meta0, meta1) -> str:
    """
    Suffix for boolean operations.
    Pattern: {idx0}{idx1}{ngon0}{ngon1}{real}3d
    Used by: cut.boolean_*

    Note: Caller must handle index canonicalization before calling.
    """
    idx0 = dtype_str(meta0.index_dtype)
    idx1 = dtype_str(meta1.index_dtype)
    return f"{idx0}{idx1}{meta0.ngon}{meta1.ngon}{dtype_str(meta0.real_dtype)}3d"
