/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (www.trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once

#include "../core/algorithm/parallel_apply.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/points_buffer.hpp"
#include "../core/vector.hpp"
#include "../core/views/zip.hpp"
#include "../topology/policy/vertex_link.hpp"

#include <algorithm>
#include <cstddef>

namespace tf {

/// @ingroup geometry_processing
/// @brief Apply Laplacian smoothing to a point set.
///
/// Iteratively moves each vertex towards the centroid of its neighbors.
/// The amount of movement is controlled by lambda (0 = no movement, 1 = full).
///
/// @param pts Point set with vertex_link policy attached.
/// @param iterations Number of smoothing iterations.
/// @param lambda Smoothing factor in [0, 1]. Default: 0.5.
/// @return New points buffer with smoothed positions.
template <typename Policy>
auto laplacian_smoothed(const tf::points<Policy> &pts, std::size_t iterations,
                        tf::coordinate_type<Policy> lambda = 0.5) {
  static_assert(tf::has_vertex_link_policy<Policy>,
                "Points must have vertex_link policy attached");

  using T = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;

  const auto &vlink = pts.vertex_link();

  tf::points_buffer<T, Dims> current;
  current.allocate(pts.size());
  tf::parallel_copy(pts, current.points());

  tf::points_buffer<T, Dims> next;
  next.allocate(pts.size());

  for (std::size_t iter = 0; iter < iterations; ++iter) {
    tf::parallel_apply(
        tf::zip(current.points(), next.points(), vlink),
        [&](auto tup) {
          auto [curr, out, neighbors] = tup;
          if (neighbors.size() == 0) {
            out = curr;
            return;
          }
          tf::vector<T, Dims> centroid{};
          for (auto n : neighbors)
            centroid += current.points()[n].as_vector_view();
          centroid /= T(neighbors.size());
          out.as_vector_view() =
              curr.as_vector_view() + (centroid - curr.as_vector_view()) * lambda;
        });
    std::swap(current, next);
  }

  return current;
}

} // namespace tf
