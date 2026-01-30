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

/** @defgroup spatial Spatial Module
 *  Trees, forms, search, neighbor queries, and ray casting.
 */

/** @defgroup spatial_structures Spatial Structures
 *  @ingroup spatial
 *  Trees and forms for organizing primitives into spatial hierarchies.
 */

/** @defgroup spatial_queries Spatial Queries
 *  @ingroup spatial
 *  Distance, intersection, neighbor search, ray casting, and custom search operations.
 */

/** @defgroup spatial_results Result Types
 *  @ingroup spatial
 *  Result structures for spatial queries.
 */

/** @defgroup spatial_configuration Configuration
 *  @ingroup spatial
 *  Tree configuration and partitioning strategies.
 */

/** @defgroup spatial_policies Policy Tags
 *  @ingroup spatial
 *  Policy-based composition for attaching spatial trees to ranges.
 */

#include "./spatial/aabb_mod_tree.hpp"         // IWYU pragma: export
#include "./spatial/aabb_tree.hpp"             // IWYU pragma: export
#include "./spatial/distance.hpp"              // IWYU pragma: export
#include "./spatial/gather_ids.hpp"            // IWYU pragma: export
#include "./spatial/gather_self_ids.hpp"       // IWYU pragma: export
#include "./spatial/intersects.hpp"            // IWYU pragma: export
#include "./spatial/mod_tree.hpp"              // IWYU pragma: export
#include "./spatial/mod_tree_like.hpp"         // IWYU pragma: export
#include "./spatial/nearest_neighbor.hpp"      // IWYU pragma: export
#include "./spatial/nearest_neighbor_pair.hpp" // IWYU pragma: export
#include "./spatial/nearest_neighbors.hpp"     // IWYU pragma: export
#include "./spatial/neighbor_search.hpp"       // IWYU pragma: export
#include "./spatial/obb_mod_tree.hpp"          // IWYU pragma: export
#include "./spatial/obb_tree.hpp"              // IWYU pragma: export
#include "./spatial/obbrss_mod_tree.hpp"       // IWYU pragma: export
#include "./spatial/obbrss_tree.hpp"           // IWYU pragma: export
#include "./spatial/partitioning.hpp"          // IWYU pragma: export
#include "./spatial/policy.hpp"                // IWYU pragma: export
#include "./spatial/ray_cast.hpp"              // IWYU pragma: export
#include "./spatial/ray_hit.hpp"               // IWYU pragma: export
#include "./spatial/search.hpp"                // IWYU pragma: export
#include "./spatial/search_self.hpp"           // IWYU pragma: export
#include "./spatial/stitch_mod_tree.hpp"       // IWYU pragma: export
#include "./spatial/tree.hpp"                  // IWYU pragma: export
#include "./spatial/tree_config.hpp"           // IWYU pragma: export
#include "./spatial/tree_index_map.hpp"        // IWYU pragma: export
#include "./spatial/tree_like.hpp"             // IWYU pragma: export
#include "./spatial/tree_metric_info.hpp"      // IWYU pragma: export
#include "./spatial/tree_metric_info_pair.hpp" // IWYU pragma: export
#include "./spatial/tree_ray_info.hpp"         // IWYU pragma: export
