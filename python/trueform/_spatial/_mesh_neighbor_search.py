"""
Dispatch table for mesh neighbor search

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .._primitives import Point, Segment, Polygon, Ray, Line

# Dispatch table for single nearest neighbor search on meshes
# Maps primitive type -> function_name_template
# Template expects suffix like "intfloat32d" (index_type, real_type, ngon, dims)
_MESH_NEIGHBOR_SEARCH_DISPATCH = {
    Point: "neighbor_search_mesh_point_{}",
    Segment: "neighbor_search_mesh_segment_{}",
    Polygon: "neighbor_search_mesh_polygon_{}",
    Ray: "neighbor_search_mesh_ray_{}",
    Line: "neighbor_search_mesh_line_{}",
}

# Dispatch table for KNN search on meshes
_MESH_NEIGHBOR_SEARCH_KNN_DISPATCH = {
    Point: "neighbor_search_mesh_knn_point_{}",
    Segment: "neighbor_search_mesh_knn_segment_{}",
    Polygon: "neighbor_search_mesh_knn_polygon_{}",
    Ray: "neighbor_search_mesh_knn_ray_{}",
    Line: "neighbor_search_mesh_knn_line_{}",
}
