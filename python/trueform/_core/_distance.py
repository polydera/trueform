"""
Internal dispatch table for distance/distance2 with core primitives

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .._primitives import Point, Segment, Polygon, Line, AABB, Ray, Plane


# Dispatch table for core distance/distance2
# Maps (type0, type1) -> (function_name_template, needs_swap)
# needs_swap=True means we need to swap arguments
# Function name template will be formatted with "distance" or "distance2" and suffix
_DISTANCE_DISPATCH = {
    # Point combinations
    (Point, Point): ("{}_point_point_{}", False),
    (Point, AABB): ("{}_point_aabb_{}", False),
    (AABB, Point): ("{}_point_aabb_{}", True),
    (Point, Plane): ("{}_point_plane_{}", False),
    (Plane, Point): ("{}_point_plane_{}", True),
    (Point, Line): ("{}_point_line_{}", False),
    (Line, Point): ("{}_point_line_{}", True),
    (Point, Ray): ("{}_point_ray_{}", False),
    (Ray, Point): ("{}_point_ray_{}", True),
    (Point, Segment): ("{}_point_segment_{}", False),
    (Segment, Point): ("{}_point_segment_{}", True),
    (Point, Polygon): ("{}_point_polygon_{}", False),
    (Polygon, Point): ("{}_point_polygon_{}", True),

    # AABB combinations
    (AABB, AABB): ("{}_aabb_aabb_{}", False),

    # Line combinations
    (Line, Line): ("{}_line_line_{}", False),
    (Line, Ray): ("{}_line_ray_{}", False),
    (Ray, Line): ("{}_line_ray_{}", True),
    (Line, Segment): ("{}_line_segment_{}", False),
    (Segment, Line): ("{}_line_segment_{}", True),
    (Line, Polygon): ("{}_line_polygon_{}", False),
    (Polygon, Line): ("{}_line_polygon_{}", True),

    # Ray combinations
    (Ray, Ray): ("{}_ray_ray_{}", False),
    (Ray, Segment): ("{}_ray_segment_{}", False),
    (Segment, Ray): ("{}_ray_segment_{}", True),
    (Ray, Polygon): ("{}_ray_polygon_{}", False),
    (Polygon, Ray): ("{}_ray_polygon_{}", True),

    # Segment combinations
    (Segment, Segment): ("{}_segment_segment_{}", False),
    (Segment, Polygon): ("{}_segment_polygon_{}", False),
    (Polygon, Segment): ("{}_segment_polygon_{}", True),

    # Polygon combinations
    (Polygon, Polygon): ("{}_polygon_polygon_{}", False),

    # Plane combinations (3D only)
    (Segment, Plane): ("{}_segment_plane_{}", False),
    (Plane, Segment): ("{}_segment_plane_{}", True),
    (Ray, Plane): ("{}_ray_plane_{}", False),
    (Plane, Ray): ("{}_ray_plane_{}", True),
    (Line, Plane): ("{}_line_plane_{}", False),
    (Plane, Line): ("{}_line_plane_{}", True),
    (Polygon, Plane): ("{}_polygon_plane_{}", False),
    (Plane, Polygon): ("{}_polygon_plane_{}", True),
    (Plane, Plane): ("{}_plane_plane_{}", False),
}
