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

/** @defgroup cpp The C++ facade
 *  A carrier assembled from the caller's own views, a cache it answers
 *  through, and one compiled body per operation.
 */

#include "./cpp/arrangement.hpp" // IWYU pragma: export
#include "./cpp/clean.hpp"       // IWYU pragma: export
#include "./cpp/core.hpp"        // IWYU pragma: export
#include "./cpp/csg.hpp"         // IWYU pragma: export
#include "./cpp/geometry.hpp"    // IWYU pragma: export
#include "./cpp/intersect.hpp"   // IWYU pragma: export
#include "./cpp/io.hpp"          // IWYU pragma: export
#include "./cpp/iso.hpp"         // IWYU pragma: export
#include "./cpp/reindex.hpp"     // IWYU pragma: export
#include "./cpp/remesh.hpp"      // IWYU pragma: export
#include "./cpp/spatial.hpp"     // IWYU pragma: export
#include "./cpp/topology.hpp"    // IWYU pragma: export
