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

/** @defgroup cpp_spatial spatial
 *  Distance, intersection, gathering, neighbours and ray casts.
 */

#include "./spatial/async/closest_metric_point.hpp"       // IWYU pragma: export
#include "./spatial/async/closest_metric_point_pair.hpp"  // IWYU pragma: export
#include "./spatial/async/distance.hpp"                   // IWYU pragma: export
#include "./spatial/async/gather_ids.hpp"                 // IWYU pragma: export
#include "./spatial/async/gather_ids_within_distance.hpp" // IWYU pragma: export
#include "./spatial/async/intersects.hpp"                 // IWYU pragma: export
#include "./spatial/async/neighbor_search.hpp"            // IWYU pragma: export
#include "./spatial/async/neighbor_search_knn.hpp"        // IWYU pragma: export
#include "./spatial/async/ray_cast.hpp"                   // IWYU pragma: export
#include "./spatial/async/signed_distance.hpp"            // IWYU pragma: export
#include "./spatial/async/transformed.hpp"                // IWYU pragma: export
#include "./spatial/closest_metric_point.hpp"             // IWYU pragma: export
#include "./spatial/closest_metric_point_pair.hpp"        // IWYU pragma: export
#include "./spatial/distance.hpp"                         // IWYU pragma: export
#include "./spatial/gather_ids.hpp"                       // IWYU pragma: export
#include "./spatial/gather_ids_within_distance.hpp"       // IWYU pragma: export
#include "./spatial/intersection_result.hpp"              // IWYU pragma: export
#include "./spatial/intersects.hpp"                       // IWYU pragma: export
#include "./spatial/neighbor_search.hpp"                  // IWYU pragma: export
#include "./spatial/neighbor_search_knn.hpp"              // IWYU pragma: export
#include "./spatial/primitive.hpp"                        // IWYU pragma: export
#include "./spatial/ray_cast.hpp"                         // IWYU pragma: export
#include "./spatial/signed_distance.hpp"                  // IWYU pragma: export
#include "./spatial/transformed.hpp"                      // IWYU pragma: export
