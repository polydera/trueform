"""
Topology operations

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .label_connected_components import label_connected_components
from .cell_membership import cell_membership
from .manifold_edge_link import manifold_edge_link
from .face_link import face_link
from .vertex_link import vertex_link_edges, vertex_link_faces
from .boundary_edges import boundary_edges
from .boundary_paths import boundary_paths
from .boundary_curves import boundary_curves
from .non_manifold_edges import non_manifold_edges

__all__ = [
    'label_connected_components',
    'cell_membership',
    'manifold_edge_link',
    'face_link',
    'vertex_link_edges',
    'vertex_link_faces',
    'boundary_edges',
    'boundary_paths',
    'boundary_curves',
    'non_manifold_edges',
]
