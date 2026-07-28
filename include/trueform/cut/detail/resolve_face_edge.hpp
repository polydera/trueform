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

#include "../../topology/is_on_same_edge.hpp"
#include "../../topology/topo_id.hpp"
#include <optional>

namespace tf::cut {

/// @ingroup cut
/// @brief Resolve which original face edge a loop edge lies on.
///
/// Given the two endpoint sub-ids of a loop edge and the original face
/// vertex count, returns the original face edge index `e` if both
/// endpoints lie on edge `e`, otherwise `std::nullopt`.
///
/// The face edges are indexed by their starting vertex (edge `e` runs
/// from corner `e` to corner `e+1 mod face_size`). When at least one
/// endpoint's `sub_id` lies in the interior of an edge (`topo_type::edge`),
/// that id IS the resolved edge. When both endpoints are corners
/// (`topo_type::vertex`), the convention assumes consistent loop
/// orientation: `s0.id` is the resolved edge id.
///
/// Per-vertex MEL indexing (using just one endpoint's `sub_id.id`) is
/// ambiguous at corners — a corner is shared by two edges. This helper
/// disambiguates by requiring both endpoints to lie on the same edge.
template <typename Index>
auto resolve_face_edge(const tf::topo_id<Index> &s0,
                       const tf::topo_id<Index> &s1, std::size_t face_size)
    -> std::optional<Index> {
  if (!tf::is_on_same_edge(s0, s1, face_size))
    return std::nullopt;
  if (s0.label == tf::topo_type::edge)
    return s0.id;
  if (s1.label == tf::topo_type::edge)
    return s1.id;
  return s0.id;
}

} // namespace tf::cut
