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
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/vertex.hpp"

#include <array>
#include <cstddef>

namespace tf::intersect {

/// The created points the rounding put on an original vertex.
///
/// A kind-E point is an exact parameter class on its home edge, and that
/// edge's two ends are lattice points already. When the blend lands on
/// one of them the class and the vertex are one point of every table
/// below, however far apart the exact producer holds them — so the pair
/// is stated here, at the rounding, and the identity gate elects it like
/// any other coincidence.
///
/// Two comparisons per class and nothing else: a class can only round
/// onto a point of its own carrier, and a vertex-anchored point is its
/// vertex already.
template <typename Index, typename Int, typename Ibp, typename GetFlatPoint>
auto find_points_on_carrier_ends(
    const Ibp &ibp, const GetFlatPoint &get_flat_point,
    const tf::buffer<tf::exact::pt3<Int>> &points,
    tf::buffer<std::array<Index, 2>> &landings) -> void {
  landings.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(ibp.n_vertex_points(), ibp.n_points()), landings,
      [&](Index id, tf::buffer<std::array<Index, 2>> &out) {
        const auto &edge = ibp.home_edge(id);
        const auto &at = points[std::size_t(id)];
        if (at == get_flat_point(edge.u))
          out.push_back({id, edge.u});
        else if (at == get_flat_point(edge.v))
          out.push_back({id, edge.v});
      },
      tf::checked(8192));
}

} // namespace tf::intersect
