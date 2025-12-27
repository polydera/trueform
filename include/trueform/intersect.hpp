/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

/** @defgroup intersect Intersect Module
 *  Mesh intersections, self-intersections, and scalar field isocontours.
 */

/** @defgroup intersect_curves Intersection Curves
 *  @ingroup intersect
 *  High-level curve extraction from mesh and scalar field intersections.
 */

/** @defgroup intersect_data Intersection Data
 *  @ingroup intersect
 *  Low-level intersection point and topology access.
 */

/** @defgroup intersect_types Intersection Types
 *  @ingroup intersect
 *  Supporting types for intersection operations.
 */

#include "./intersect/intersections_between_polygons.hpp" // IWYU pragma: export
#include "./intersect/intersections_within_polygons.hpp"  // IWYU pragma: export
#include "./intersect/intersections_within_segments.hpp"  // IWYU pragma: export
#include "./intersect/make_intersection_curves.hpp"       // IWYU pragma: export
#include "./intersect/make_intersection_edges.hpp"        // IWYU pragma: export
#include "./intersect/make_isocurves.hpp"                 // IWYU pragma: export
#include "./intersect/make_self_intersection_curves.hpp"  // IWYU pragma: export
#include "./intersect/scalar_field_intersections.hpp"     // IWYU pragma: export
