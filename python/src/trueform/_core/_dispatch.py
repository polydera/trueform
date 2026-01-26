"""
Dispatch tables for tf::core operations (primitive x primitive).

All share the same suffix pattern: {real}{dims}d

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from .._primitives import Point, Segment, Polygon, Line, AABB, Ray, Plane


# =============================================================================
# INTERSECTS: (type0, type1) -> (func_template, needs_swap)
# =============================================================================
INTERSECTS = {
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


# =============================================================================
# DISTANCE: (type0, type1) -> (func_base_template, needs_swap)
# func_base_template has {} for "distance" or "distance2", then {} for suffix
# =============================================================================
DISTANCE = {
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


# =============================================================================
# RAY_CAST: target_type -> func_template
# Ray is always the source, no needs_swap
# =============================================================================
RAY_CAST = {
    Plane: "ray_cast_ray_plane_{}",
    Polygon: "ray_cast_ray_polygon_{}",
    Segment: "ray_cast_ray_segment_{}",
    Line: "ray_cast_ray_line_{}",
    AABB: "ray_cast_ray_aabb_{}",
}
