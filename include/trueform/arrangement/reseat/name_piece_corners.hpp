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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "./reseat_pieces.hpp"
#include <array>
#include <cstddef>

namespace tf::arrangement::reseat {

/// Every piece's corners in the names a layer states now (`target`, as
/// @ref tf::intersect::graph::plane_targeted_endpoint reads it): the corners
/// a weld folds are one corner.
template <typename Index, typename Target, typename VertexOffsets>
auto name_piece_corners(
    const Target &target, const VertexOffsets &vertex_offsets,
    const tf::buffer<std::array<piece_corner<Index>, 3>> &corners,
    tf::buffer<std::array<piece_corner<Index>, 3>> &out) -> void {
  out.allocate(corners.size());
  tf::parallel_for_each(
      tf::make_sequence_range(corners.size()),
      [&](std::size_t k) {
        for (std::size_t c = 0; c < 3; ++c)
          tf::intersect::graph::plane_targeted_endpoint(
              target, vertex_offsets, corners[k][c].tag, corners[k][c].id,
              out[k][c].tag, out[k][c].id);
      },
      tf::checked);
}

} // namespace tf::arrangement::reseat
