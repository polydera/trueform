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
#include "../core/none.hpp"
#include "./face_membership.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Create face membership from polygons.
///
/// Convenience function that builds and returns a face_membership structure.
/// The index type is automatically deduced from the polygons unless specified.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A face_membership structure mapping vertices to faces.
template <typename Index = tf::none_t, typename Policy>
auto make_face_membership(const tf::polygons<Policy> &polygons) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_face_membership<std::decay_t<decltype(polygons.faces()[0][0])>>(
        polygons);
  } else {
    tf::face_membership<Index> fm;
    fm.build(polygons);
    return fm;
  }
}

} // namespace tf
