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

/** @defgroup cpp_reindex reindex
 *  Selection, concatenation and splitting of carriers.
 */

#include "./reindex/async/concatenated.hpp"          // IWYU pragma: export
#include "./reindex/async/mesh.hpp"                  // IWYU pragma: export
#include "./reindex/async/selections.hpp"            // IWYU pragma: export
#include "./reindex/async/split_into_components.hpp" // IWYU pragma: export
#include "./reindex/async/split_into_domains.hpp"    // IWYU pragma: export
#include "./reindex/by_ids.hpp"                      // IWYU pragma: export
#include "./reindex/by_ids_on_points.hpp"            // IWYU pragma: export
#include "./reindex/by_mask.hpp"                     // IWYU pragma: export
#include "./reindex/by_mask_on_points.hpp"           // IWYU pragma: export
#include "./reindex/concatenated.hpp"                // IWYU pragma: export
#include "./reindex/fixed_arity.hpp"                 // IWYU pragma: export
#include "./reindex/mesh.hpp"                        // IWYU pragma: export
#include "./reindex/selection_result.hpp"            // IWYU pragma: export
#include "./reindex/split_into_components.hpp"       // IWYU pragma: export
#include "./reindex/split_into_domains.hpp"          // IWYU pragma: export
