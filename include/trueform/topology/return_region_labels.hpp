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

namespace tf {

/// @ingroup topology
/// @brief Tag type to request region-label return.
///
/// Pass @ref tf::return_region_labels to @ref tf::make_cdt to receive the
/// unfiltered triangulation together with its per-triangle region labels.
struct return_region_labels_t {};

/// @ingroup topology
/// @brief Tag instance for requesting region labels.
inline constexpr return_region_labels_t return_region_labels{};

} // namespace tf
