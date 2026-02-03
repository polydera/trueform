"""
Topology operations

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .label_connected_components import label_connected_components
from .cell_membership import cell_membership
from .manifold_edge_link import manifold_edge_link
from .face_link import face_link
from .vertex_link import vertex_link_edges, vertex_link_faces
from .k_rings import k_rings
from .neighborhoods import neighborhoods
from .boundary_edges import boundary_edges
from .boundary_paths import boundary_paths
from .boundary_curves import boundary_curves
from .non_manifold_edges import non_manifold_edges
from .orient_faces_consistently import orient_faces_consistently
from .connect_edges_to_paths import connect_edges_to_paths
from .is_closed import is_closed, is_open
from .is_manifold import is_manifold, is_non_manifold

__all__ = [
    'label_connected_components',
    'cell_membership',
    'manifold_edge_link',
    'face_link',
    'vertex_link_edges',
    'vertex_link_faces',
    'k_rings',
    'neighborhoods',
    'boundary_edges',
    'boundary_paths',
    'boundary_curves',
    'non_manifold_edges',
    'orient_faces_consistently',
    'connect_edges_to_paths',
    'is_closed',
    'is_open',
    'is_manifold',
    'is_non_manifold',
]
