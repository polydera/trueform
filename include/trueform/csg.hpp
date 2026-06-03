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

/** @defgroup csg CSG Module
 *  Implicit N-form CSG: build an arrangement once for a set of
 *  forms (@ref tf::csg_graph), then evaluate any boolean expression
 *  over the operand bits (@ref tf::csg::expr) to extract the result
 *  mesh (@ref tf::make_csg_mesh).
 */

#include "./csg/csg_graph.hpp"      // IWYU pragma: export
#include "./csg/expression.hpp"     // IWYU pragma: export
#include "./csg/graph.hpp"          // IWYU pragma: export
#include "./csg/make_csg_graph.hpp" // IWYU pragma: export
#include "./csg/make_csg_mesh.hpp"  // IWYU pragma: export
