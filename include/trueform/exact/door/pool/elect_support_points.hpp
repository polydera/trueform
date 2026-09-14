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

#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/offset_block_buffer.hpp"
#include "../../../core/point.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../meta.hpp"
#include "../placement_tables.hpp"

#include <cstddef>

namespace tf::exact::door::pool {

/// The original vertex each name is described from: the least corner of the
/// name's own supported faces by (squared lattice norm, lexicographic
/// coordinates).
///
/// FEATURE CORNERS COUNT HERE. The nonfeature support of a valid name can be
/// empty — every face of it may carry a feature corner — and the ray the
/// pool is described along still has to pass through a point the geometry
/// actually stood on.
template <typename Index, typename Int, typename RealType>
auto elect_support_points(const placement_tables<Index, Int, RealType> &tables,
                          const tf::blocked_buffer<Index, 3> &corners,
                          const tf::offset_block_buffer<int, Index> &name_faces,
                          tf::buffer<tf::point<Int, 3>> &support_point)
    -> void {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto square_norm = [](const tf::point<Int, 3> &p) {
    return T2(p[0]) * T2(p[0]) + T2(p[1]) * T2(p[1]) + T2(p[2]) * T2(p[2]);
  };

  support_point.allocate(name_faces.size());
  tf::parallel_for_each(
      tf::make_sequence_range(name_faces.size()),
      [&tables, &corners, &name_faces, &support_point,
       square_norm](std::size_t n) {
        bool elected = false;
        T2 held(0);
        for (const auto face : name_faces[n])
          for (const auto v : corners[std::size_t(face)]) {
            const auto &corner = tables.points[std::size_t(v)];
            const auto offered = square_norm(corner);
            if (!elected || offered < held ||
                (offered == held && corner < support_point[n])) {
              support_point[n] = corner;
              held = offered;
              elected = true;
            }
          }
      },
      tf::checked);
}

} // namespace tf::exact::door::pool
