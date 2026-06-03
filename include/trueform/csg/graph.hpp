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

/** @defgroup csg_graph_internals CSG Graph Internals
 *  @ingroup csg
 *  Low-level primitives used by @ref tf::make_csg_mesh: per-domain
 *  evaluation, per-component side selection, partitioning, vertex
 *  remap, and the underlying mesh extractor.
 */

#include "./graph/compute_chosen_sides.hpp" // IWYU pragma: export
#include "./graph/evaluate_per_domain.hpp"  // IWYU pragma: export
#include "./graph/make_csg_map_data.hpp"    // IWYU pragma: export
#include "./graph/make_csg_mesh.hpp"        // IWYU pragma: export
#include "./graph/make_csg_partition.hpp"   // IWYU pragma: export
