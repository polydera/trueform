/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once

/** @defgroup intersect Intersect Module
 *  Mesh intersections and self-intersections.
 */

/** @defgroup intersect_curves Intersection Curves
 *  @ingroup intersect
 *  High-level curve extraction from mesh intersections.
 */

/** @defgroup intersect_data Intersection Data
 *  @ingroup intersect
 *  Low-level intersection point and topology access.
 */

#include "./intersect/intersections_within_segments.hpp" // IWYU pragma: export
#include "./intersect/make_intersection_curves.hpp"      // IWYU pragma: export
#include "./intersect/make_intersection_edges.hpp"       // IWYU pragma: export
#include "./intersect/make_self_intersection_curves.hpp" // IWYU pragma: export
