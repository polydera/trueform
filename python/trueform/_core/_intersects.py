"""
Internal dispatch table for intersects with core primitives

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .._primitives import Point, Segment, Polygon, Line, AABB, Ray, Plane


# Dispatch table for core intersects
# Maps (type0, type1) -> (function_name_template, needs_swap)
# needs_swap=True means we need to swap arguments
_INTERSECTS_DISPATCH = {
    # Point combinations
    (Point, Point): ("intersects_point_point_{}", False),
    (Point, AABB): ("intersects_point_aabb_{}", False),
    (AABB, Point): ("intersects_point_aabb_{}", True),
    (Point, Line): ("intersects_point_line_{}", False),
    (Line, Point): ("intersects_point_line_{}", True),
    (Point, Ray): ("intersects_point_ray_{}", False),
    (Ray, Point): ("intersects_point_ray_{}", True),
    (Point, Segment): ("intersects_point_segment_{}", False),
    (Segment, Point): ("intersects_point_segment_{}", True),
    (Point, Polygon): ("intersects_point_polygon_{}", False),
    (Polygon, Point): ("intersects_point_polygon_{}", True),
    (Point, Plane): ("intersects_point_plane_{}", False),
    (Plane, Point): ("intersects_point_plane_{}", True),

    # AABB combinations
    (AABB, AABB): ("intersects_aabb_aabb_{}", False),
    (AABB, Plane): ("intersects_aabb_plane_{}", False),
    (Plane, AABB): ("intersects_aabb_plane_{}", True),
    (Segment, AABB): ("intersects_segment_aabb_{}", False),
    (AABB, Segment): ("intersects_segment_aabb_{}", True),
    (Ray, AABB): ("intersects_ray_aabb_{}", False),
    (AABB, Ray): ("intersects_ray_aabb_{}", True),
    (Line, AABB): ("intersects_line_aabb_{}", False),
    (AABB, Line): ("intersects_line_aabb_{}", True),
    (Polygon, AABB): ("intersects_polygon_aabb_{}", False),
    (AABB, Polygon): ("intersects_polygon_aabb_{}", True),

    # Line combinations
    (Line, Line): ("intersects_line_line_{}", False),
    (Line, Ray): ("intersects_line_ray_{}", False),
    (Ray, Line): ("intersects_line_ray_{}", True),
    (Line, Segment): ("intersects_line_segment_{}", False),
    (Segment, Line): ("intersects_line_segment_{}", True),
    (Line, Polygon): ("intersects_line_polygon_{}", False),
    (Polygon, Line): ("intersects_line_polygon_{}", True),
    (Line, Plane): ("intersects_line_plane_{}", False),
    (Plane, Line): ("intersects_line_plane_{}", True),

    # Ray combinations
    (Ray, Ray): ("intersects_ray_ray_{}", False),
    (Ray, Segment): ("intersects_ray_segment_{}", False),
    (Segment, Ray): ("intersects_ray_segment_{}", True),
    (Ray, Polygon): ("intersects_ray_polygon_{}", False),
    (Polygon, Ray): ("intersects_ray_polygon_{}", True),
    (Ray, Plane): ("intersects_ray_plane_{}", False),
    (Plane, Ray): ("intersects_ray_plane_{}", True),

    # Segment combinations
    (Segment, Segment): ("intersects_segment_segment_{}", False),
    (Segment, Polygon): ("intersects_segment_polygon_{}", False),
    (Polygon, Segment): ("intersects_segment_polygon_{}", True),
    (Segment, Plane): ("intersects_segment_plane_{}", False),
    (Plane, Segment): ("intersects_segment_plane_{}", True),

    # Polygon combinations
    (Polygon, Polygon): ("intersects_polygon_polygon_{}", False),
    (Polygon, Plane): ("intersects_polygon_plane_{}", False),
    (Plane, Polygon): ("intersects_polygon_plane_{}", True),
    # Plane combinations
    (Plane, Plane): ("intersects_plane_plane_{}", False),
}
