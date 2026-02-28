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

/// @ingroup cut_boolean
/// @brief Configuration for boolean operations.
struct boolean_config {
  /// When true, uses ray-based containment through joint components
  /// where open mesh components are treated as outside.
  /// When false, uses pseudonormal signed distance.
  /// Default: false.
  bool everything_is_outside_of_an_open_mesh = false;
};

} // namespace tf
