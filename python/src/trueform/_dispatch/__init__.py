"""
Dispatch utilities for C++ function lookup.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .meta import InputMeta, extract_meta
from .suffix import dtype_str, build_suffix, build_suffix_pair, topology_suffix, connectivity_suffix
from .canonicalize import canonicalize_index_order
from .ensure_mesh import ensure_mesh
from .ensure_point_cloud import ensure_point_cloud

__all__ = [
    'InputMeta',
    'extract_meta',
    'dtype_str',
    'build_suffix',
    'build_suffix_pair',
    'topology_suffix',
    'connectivity_suffix',
    'canonicalize_index_order',
    'ensure_mesh',
    'ensure_point_cloud',
]
