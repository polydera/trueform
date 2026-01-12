"""
Type metadata extraction - Python equivalent of tf::coordinate_type deduction.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class FormMeta:
    """Metadata extracted from Mesh, EdgeMesh, PointCloud."""
    index_dtype: Optional[np.dtype]  # None for PointCloud
    real_dtype: np.dtype
    ngon: Optional[str]              # "2", "3", "dyn", or None for PointCloud
    dims: int


@dataclass(frozen=True)
class PrimitiveMeta:
    """Metadata extracted from Point, Segment, Polygon, etc."""
    real_dtype: np.dtype
    dims: int


def extract_form_meta(form: Any) -> FormMeta:
    """
    Extract type metadata from a spatial form.

    Recursively deduces types like C++ coordinate_type_deducer.
    """
    # Import locally to avoid cycles
    from .._spatial import Mesh, EdgeMesh, PointCloud

    if isinstance(form, Mesh):
        faces = form.faces
        index_dtype = faces.dtype if hasattr(faces, 'dtype') else faces.data.dtype
        return FormMeta(
            index_dtype=index_dtype,
            real_dtype=form.points.dtype,
            ngon='dyn' if form.is_dynamic else str(form.ngon),
            dims=form.dims
        )
    elif isinstance(form, EdgeMesh):
        return FormMeta(
            index_dtype=form.edges.dtype,
            real_dtype=form.points.dtype,
            ngon='2',
            dims=form.dims
        )
    elif isinstance(form, PointCloud):
        return FormMeta(
            index_dtype=None,
            real_dtype=form.points.dtype,
            ngon=None,
            dims=form.dims
        )
    raise TypeError(f"Unknown form: {type(form).__name__}")


def extract_primitive_meta(prim: Any) -> PrimitiveMeta:
    """Extract type metadata from a primitive."""
    return PrimitiveMeta(real_dtype=prim.dtype, dims=prim.dims)
