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

/// @brief Per-cleaning-call configuration: tolerance + which optional
/// passes run. Same struct works for polygons (primitive = face) and
/// segments (primitive = edge). Degenerate primitives are always
/// removed (they aren't legitimate geometry).
template <typename Tolerance> struct clean_config_t {
  Tolerance tolerance{};
  bool remove_duplicate_primitives = true;
  bool remove_unreferenced_points = true;
};

/// @brief Factory for @ref tf::clean_config_t with defaults.
template <typename Tolerance = int>
auto clean_config(Tolerance tolerance = Tolerance{},
                  bool remove_duplicate_primitives = true,
                  bool remove_unreferenced_points = true)
    -> clean_config_t<Tolerance> {
  return {tolerance, remove_duplicate_primitives, remove_unreferenced_points};
}

} // namespace tf
