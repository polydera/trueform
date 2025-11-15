"""
Dispatch table for point cloud neighbor search

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .._primitives import Point, Segment, Polygon, Ray, Line

# Dispatch table for single nearest neighbor search on point clouds
# Maps primitive type -> function_name_template
_POINT_CLOUD_NEIGHBOR_SEARCH_DISPATCH = {
    Point: "neighbor_search_point_{}",
    Segment: "neighbor_search_segment_{}",
    Polygon: "neighbor_search_polygon_{}",
    Ray: "neighbor_search_ray_{}",
    Line: "neighbor_search_line_{}",
}

# Dispatch table for KNN search on point clouds
_POINT_CLOUD_NEIGHBOR_SEARCH_KNN_DISPATCH = {
    Point: "neighbor_search_knn_point_{}",
    Segment: "neighbor_search_knn_segment_{}",
    Polygon: "neighbor_search_knn_polygon_{}",
    Ray: "neighbor_search_knn_ray_{}",
    Line: "neighbor_search_knn_line_{}",
}
