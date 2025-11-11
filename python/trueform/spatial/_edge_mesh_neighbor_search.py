"""
Dispatch table for edge mesh neighbor search

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from ..primitives import Point, Segment, Polygon, Ray, Line

# Dispatch table for single nearest neighbor search on edge meshes
# Maps primitive type -> function_name_template
# Template expects suffix like "intfloat2d" (index_type, real_type, dims)
_EDGE_MESH_NEIGHBOR_SEARCH_DISPATCH = {
    Point: "neighbor_search_edge_mesh_point_{}",
    Segment: "neighbor_search_edge_mesh_segment_{}",
    Polygon: "neighbor_search_edge_mesh_polygon_{}",
    Ray: "neighbor_search_edge_mesh_ray_{}",
    Line: "neighbor_search_edge_mesh_line_{}",
}

# Dispatch table for KNN search on edge meshes
_EDGE_MESH_NEIGHBOR_SEARCH_KNN_DISPATCH = {
    Point: "neighbor_search_edge_mesh_knn_point_{}",
    Segment: "neighbor_search_edge_mesh_knn_segment_{}",
    Polygon: "neighbor_search_edge_mesh_knn_polygon_{}",
    Ray: "neighbor_search_edge_mesh_knn_ray_{}",
    Line: "neighbor_search_edge_mesh_knn_line_{}",
}
