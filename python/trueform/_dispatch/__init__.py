"""
Dispatch utilities for C++ function lookup.

Provides type metadata extraction and suffix building, mirroring C++ template
dispatch patterns like tf::coordinate_type.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

from .types import (
    FormMeta,
    PrimitiveMeta,
    extract_form_meta,
    extract_primitive_meta,
)

from .suffix import (
    dtype_str,
    primitive_suffix,
    form_primitive_suffix,
    form_form_suffix,
    indexed_geometry_suffix,
    points_suffix,
    reindex_points_suffix,
    soup_suffix,
    topology_suffix,
    topology_mesh_suffix,
    isocontour_suffix,
    boolean_suffix,
)

from .canonicalize import canonicalize_index_order

__all__ = [
    # Types
    'FormMeta',
    'PrimitiveMeta',
    'extract_form_meta',
    'extract_primitive_meta',
    # Suffix builders
    'dtype_str',
    'primitive_suffix',
    'form_primitive_suffix',
    'form_form_suffix',
    'indexed_geometry_suffix',
    'points_suffix',
    'reindex_points_suffix',
    'soup_suffix',
    'topology_suffix',
    'topology_mesh_suffix',
    'isocontour_suffix',
    'boolean_suffix',
    # Canonicalization
    'canonicalize_index_order',
]
