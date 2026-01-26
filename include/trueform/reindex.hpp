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

/** @defgroup reindex Reindex Module
 *  Apply index maps and filter/combine geometry.
 */

#include "./reindex/by_ids.hpp"                // IWYU pragma: export
#include "./reindex/by_ids_on_points.hpp"      // IWYU pragma: export
#include "./reindex/by_mask.hpp"               // IWYU pragma: export
#include "./reindex/by_mask_on_points.hpp"     // IWYU pragma: export
#include "./reindex/concatenated.hpp"          // IWYU pragma: export
#include "./reindex/points.hpp"                // IWYU pragma: export
#include "./reindex/polygons.hpp"              // IWYU pragma: export
#include "./reindex/range.hpp"                 // IWYU pragma: export
#include "./reindex/return_index_map.hpp"      // IWYU pragma: export
#include "./reindex/segments.hpp"              // IWYU pragma: export
#include "./reindex/split_into_components.hpp" // IWYU pragma: export
#include "./reindex/unit_vectors.hpp"          // IWYU pragma: export
#include "./reindex/vectors.hpp"               // IWYU pragma: export
