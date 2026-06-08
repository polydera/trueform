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

/** @defgroup remesh Remesh Module
 *  Edge collapse, decimation policies, and mesh simplification.
 */

#include "./remesh/collapse_checker.hpp"         // IWYU pragma: export
#include "./remesh/collapse_config.hpp"          // IWYU pragma: export
#include "./remesh/collapse_guard_config.hpp"  // IWYU pragma: export
#include "./remesh/collapse_edges.hpp"           // IWYU pragma: export
#include "./remesh/collapse_handler.hpp"         // IWYU pragma: export
#include "./remesh/collapse_short_edges.hpp"     // IWYU pragma: export
#include "./remesh/collapsed_short_edges.hpp"    // IWYU pragma: export
#include "./remesh/decimate.hpp"                 // IWYU pragma: export
#include "./remesh/decimated.hpp"                // IWYU pragma: export
#include "./remesh/error_remesh.hpp"             // IWYU pragma: export
#include "./remesh/flip_min_angle.hpp"           // IWYU pragma: export
#include "./remesh/improve_config.hpp"           // IWYU pragma: export
#include "./remesh/improve_triangulation.hpp"    // IWYU pragma: export
#include "./remesh/isotropic_remesh.hpp"         // IWYU pragma: export
#include "./remesh/isotropic_remeshed.hpp"       // IWYU pragma: export
#include "./remesh/optimize_valence.hpp"         // IWYU pragma: export
#include "./remesh/simplify.hpp"                 // IWYU pragma: export
#include "./remesh/simplify_config.hpp"          // IWYU pragma: export
#include "./remesh/simplified.hpp"               // IWYU pragma: export
#include "./remesh/split_edges.hpp"              // IWYU pragma: export
#include "./remesh/split_handler.hpp"            // IWYU pragma: export
#include "./remesh/split_long_edges.hpp"         // IWYU pragma: export
