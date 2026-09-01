"""
CSG graph: build one arrangement, answer many boolean queries.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from .boolean import boolean_union, boolean_intersection, boolean_difference
from .csg_graph import CsgGraph
from .expr import Expr, op
from .index_maps import DomainsIndexMap, MeshIndexMap
from .outer_shell import outer_shell

__all__ = ["CsgGraph", "DomainsIndexMap", "Expr", "MeshIndexMap",
           "boolean_difference", "boolean_intersection", "boolean_union",
           "op", "outer_shell"]
