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

namespace tf {

/// @ingroup fill
/// @brief How one @ref tf::fill_holes run shapes the patches it states.
///
/// Both requests reach every group alike; the work bounds itself.
struct hole_fill_config {
  /// The quality floor the refining tier aims its patch at, in
  /// @ref tf::triangle_quality units. Zero asks for no refinement.
  double min_quality = 0.3;
  /// Whether a refined patch is then faired against its own host
  /// neighbourhood. Only the points a patch minted strictly inside itself
  /// ever move.
  bool fairing = true;
};

} // namespace tf
