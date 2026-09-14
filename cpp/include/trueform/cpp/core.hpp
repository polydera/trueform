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

/** @defgroup cpp_core core
 *  The layer's carriers — arrays, meshes, edge meshes, point clouds —
 *  and the caches they answer through.
 */

#include "./core/async/completion.hpp"         // IWYU pragma: export
#include "./core/async/future_state.hpp"       // IWYU pragma: export
#include "./core/async/histogram.hpp"          // IWYU pragma: export
#include "./core/async/nd_array_creation.hpp"  // IWYU pragma: export
#include "./core/async/nd_array_indexing.hpp"  // IWYU pragma: export
#include "./core/async/nd_array_sorting.hpp"   // IWYU pragma: export
#include "./core/async/nd_array_structure.hpp" // IWYU pragma: export
#include "./core/async/obb.hpp"                // IWYU pragma: export
#include "./core/async/reductions.hpp"         // IWYU pragma: export
#include "./core/async/submit.hpp"             // IWYU pragma: export
#include "./core/build_edge_membership.hpp"    // IWYU pragma: export
#include "./core/build_face_link.hpp"          // IWYU pragma: export
#include "./core/build_face_membership.hpp"    // IWYU pragma: export
#include "./core/build_face_normals.hpp"       // IWYU pragma: export
#include "./core/build_half_edges.hpp"         // IWYU pragma: export
#include "./core/build_manifold_edge_link.hpp" // IWYU pragma: export
#include "./core/build_point_normals.hpp"      // IWYU pragma: export
#include "./core/build_tree.hpp"               // IWYU pragma: export
#include "./core/build_vertex_link.hpp"        // IWYU pragma: export
#include "./core/build_winding_moments.hpp"    // IWYU pragma: export
#include "./core/cache.hpp"                    // IWYU pragma: export
#include "./core/common_index.hpp"             // IWYU pragma: export
#include "./core/edge_mesh.hpp"                // IWYU pragma: export
#include "./core/edge_mesh_cache.hpp"          // IWYU pragma: export
#include "./core/edge_mesh_geometry.hpp"       // IWYU pragma: export
#include "./core/elementwise.hpp"              // IWYU pragma: export
#include "./core/histogram.hpp"                // IWYU pragma: export
#include "./core/identity_transformation.hpp"  // IWYU pragma: export
#include "./core/index_map.hpp"                // IWYU pragma: export
#include "./core/index_type.hpp"               // IWYU pragma: export
#include "./core/mesh.hpp"                     // IWYU pragma: export
#include "./core/mesh_geometry.hpp"            // IWYU pragma: export
#include "./core/nd_array.hpp"                 // IWYU pragma: export
#include "./core/nd_array_creation.hpp"        // IWYU pragma: export
#include "./core/nd_array_indexing.hpp"        // IWYU pragma: export
#include "./core/nd_array_sorting.hpp"         // IWYU pragma: export
#include "./core/nd_array_structure.hpp"       // IWYU pragma: export
#include "./core/obb.hpp"                      // IWYU pragma: export
#include "./core/offset_blocked_buffer.hpp"    // IWYU pragma: export
#include "./core/parallel_config.hpp"          // IWYU pragma: export
#include "./core/point_cloud.hpp"              // IWYU pragma: export
#include "./core/point_cloud_cache.hpp"        // IWYU pragma: export
#include "./core/point_cloud_geometry.hpp"     // IWYU pragma: export
#include "./core/reductions.hpp"               // IWYU pragma: export
#include "./core/to_matrix.hpp"                // IWYU pragma: export
