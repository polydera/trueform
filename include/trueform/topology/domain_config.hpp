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

/// @ingroup topology_components
/// @brief Flags controlling @ref tf::make_domain_labels behaviour.
enum class domain_config : int {
  /// No flags set.
  none = 0,
  /// Drop the unbounded outer domain from the output; only bounded
  /// domains receive ids.
  exclude_outer_shell = 1,
  /// Mask MEL components carrying boundary edges out of the reduced
  /// graph (Mode 1). When unset, the two sides of each such component
  /// are fused at the pre-merge (Mode 2).
  ignore_open_fragments = 2,
};

constexpr auto operator|(domain_config a, domain_config b) -> domain_config {
  return static_cast<domain_config>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr auto operator&(domain_config a, domain_config b) -> bool {
  return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

} // namespace tf
