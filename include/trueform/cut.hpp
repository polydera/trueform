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

/** @defgroup cut Cut Module
 *  Mesh booleans, isocurve embedding, and planar arrangements.
 */

/** @defgroup cut_isocurves Isocurve Embedding
 *  @ingroup cut
 *  Embedding scalar field isocurves and isobands into mesh topology.
 */

/** @defgroup cut_boolean Boolean Operations
 *  @ingroup cut
 *  Mesh boolean operations and arrangements.
 */

/** @defgroup cut_planar Planar Arrangements
 *  @ingroup cut
 *  2D segment arrangements and overlays.
 */

/** @defgroup cut_types Cut Types
 *  @ingroup cut
 *  Supporting types for cut operations.
 */

/** @defgroup cut_data Cut Data
 *  @ingroup cut
 *  Low-level face cutting infrastructure.
 */

#include "./cut/arrangement_class.hpp"                 // IWYU pragma: export
#include "./cut/cut_faces.hpp"                         // IWYU pragma: export
#include "./cut/embedded_isocurves.hpp"                // IWYU pragma: export
#include "./cut/embedded_self_intersection_curves.hpp" // IWYU pragma: export
#include "./cut/loops.hpp"                             // IWYU pragma: export
#include "./cut/make_boolean.hpp"                      // IWYU pragma: export
#include "./cut/make_boolean_pair.hpp"                 // IWYU pragma: export
#include "./cut/make_isobands.hpp"                     // IWYU pragma: export
#include "./cut/make_mesh_arrangements.hpp"            // IWYU pragma: export
#include "./cut/planar_arrangements.hpp"               // IWYU pragma: export
#include "./cut/planar_overlay.hpp"                    // IWYU pragma: export
#include "./cut/scalar_cut_faces.hpp"                  // IWYU pragma: export
#include "./cut/tagged_cut_faces.hpp"                  // IWYU pragma: export
