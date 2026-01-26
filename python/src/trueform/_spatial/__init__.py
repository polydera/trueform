"""
Spatial query operations

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

# Spatial forms
from .mesh import Mesh
from .edge_mesh import EdgeMesh
from .point_cloud import PointCloud
from .gather_ids import gather_intersecting_ids, gather_ids_within_distance
from .neighbor_search import neighbor_search


__all__ = ['Mesh', 'EdgeMesh', 'PointCloud', 'neighbor_search', 'gather_intersecting_ids', 'gather_ids_within_distance']
