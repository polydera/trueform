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

namespace tf::csg {

/// @ingroup csg_expression
/// @brief Which pieces of a selected surface an expression emits.
enum class selection_kind : unsigned char {
  /// The pieces bounding the expression's region: exactly one side
  /// satisfies it. Wound outward from the region.
  boundary,
  /// The pieces lying inside the region: both sides satisfy it. The
  /// form's stored winding.
  inside,
};

} // namespace tf::csg
