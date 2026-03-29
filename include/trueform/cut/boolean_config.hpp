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
  /// When true, uses ray-based containment to correctly classify
  /// multi-nested geometry (e.g., a shell inside another shell).
  /// Open mesh components are treated as having no interior.
  /// When false, uses signed distance to the nearest surface —
  /// faster but only sees the closest component.
  bool support_multi_nesting = true;
};

} // namespace tf
