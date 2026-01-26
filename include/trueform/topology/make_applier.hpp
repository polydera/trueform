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

#include "../core/range.hpp"
#include "./face_link_like.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./vertex_link_like.hpp"

namespace tf {

/// @ingroup topology_components
/// @brief Create an applier from a face link structure.
///
/// Returns a function that iterates over face neighbors, suitable for
/// use with @ref tf::label_connected_components().
///
/// @tparam Policy The face link policy type.
/// @param link The face link structure.
/// @return An applier function.
template <typename Policy>
auto make_applier(const tf::face_link_like<Policy> &link) {
  return [link = tf::make_range(link)](auto id, const auto &f) {
    for (auto n_id : link[id])
      f(n_id);
  };
}
/// @ingroup topology_components
/// @brief Create an applier from a vertex link structure.
///
/// Returns a function that iterates over vertex neighbors, suitable for
/// use with @ref tf::label_connected_components().
///
/// @tparam Policy The vertex link policy type.
/// @param link The vertex link structure.
/// @return An applier function.
template <typename Policy>
auto make_applier(const tf::vertex_link_like<Policy> &link) {
  return [link = tf::make_range(link)](auto id, const auto &f) {
    for (auto n_id : link[id])
      f(n_id);
  };
}

/// @ingroup topology_components
/// @brief Create an applier from a manifold edge link structure.
///
/// Returns a function that iterates over simple (manifold) edge neighbors,
/// suitable for use with @ref tf::label_connected_components().
/// Non-manifold and boundary edges are skipped.
///
/// @tparam Policy The manifold edge link policy type.
/// @param link The manifold edge link structure.
/// @return An applier function.
template <typename Policy>
auto make_applier(const tf::manifold_edge_link_like<Policy> &link) {
  return [link = tf::make_range(link)](auto id, const auto &f) {
    for (const auto &he : link[id])
      if (he.is_simple())
        f(he.face_peer);
  };
}
} // namespace tf
