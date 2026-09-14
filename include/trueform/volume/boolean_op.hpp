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

/// @ingroup volume
/// @brief Constructive-solid-geometry operation on two signed distance fields.
enum class volume_boolean_op {
  union_,       ///< A ∪ B — the region inside either field.
  intersection, ///< A ∩ B — the region inside both fields.
  difference,   ///< A \ B — the region inside A but not B.
};

} // namespace tf
