/*
 * Copyright (c) 2026 XLAB
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

/** @defgroup arrangement Arrangement Module
 *  Mesh, polygon, and segment arrangements and their reads.
 */

/** @defgroup arrangement_mesh Mesh Arrangements
 *  @ingroup arrangement
 *  Mesh arrangements: the classified regions of a set of forms.
 */

/** @defgroup arrangement_planar Planar Arrangements
 *  @ingroup arrangement
 *  2D segment arrangements and overlays.
 */

#include "./arrangement/arrangement_graph.hpp"         // IWYU pragma: export
#include "./arrangement/make_arrangement_graph.hpp"    // IWYU pragma: export
#include "./arrangement/make_arrangement_mesh.hpp"     // IWYU pragma: export
#include "./arrangement/make_intersection_curves.hpp"  // IWYU pragma: export
#include "./arrangement/make_mesh_arrangements.hpp"    // IWYU pragma: export
#include "./arrangement/make_polygon_arrangements.hpp" // IWYU pragma: export
#include "./arrangement/make_segment_arrangements.hpp" // IWYU pragma: export
#include "./arrangement/make_stitch_index_map.hpp"     // IWYU pragma: export
