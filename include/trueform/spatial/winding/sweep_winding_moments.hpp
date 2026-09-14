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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./raw_winding_moment.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::spatial {

/// Sum each inner node's moment from its children, bottom-up by level.
/// The implicit heap makes each level a closed-form index range
/// (start' = inner_size * start + 1) and states the arity on every
/// inner node (first_child = inner_size * id + 1), every child lives
/// one level down, so the levels run deepest-first and the nodes of one
/// level in parallel; empty children are zero rows and add nothing.
/// A child's second moment sits about its own centroid; the parent
/// composes it by the shift theorem — the child's dipole times its
/// centroid's offset from the parent's — so every row stays stated
/// about its own centroid and no scale ever meets the mesh's absolute
/// coordinates.
template <typename Nodes>
auto sweep_winding_moments(raw_winding_moment *moments, const Nodes &nodes)
    -> void {
  std::size_t inner_size = nodes.size();
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    if (node.is_leaf() || node.is_empty())
      continue;
    inner_size = (std::size_t(node.get_data()[0]) - 1) / i;
    break;
  }
  tf::small_vector<std::size_t, 64> starts;
  starts.push_back(0);
  while (starts.back() < nodes.size())
    starts.push_back(std::size_t(inner_size) * starts.back() + 1);
  for (std::size_t level = starts.size() - 1; level-- > 0;) {
    const auto lo = starts[level];
    const auto hi = std::min(starts[level + 1], std::size_t(nodes.size()));
    tf::parallel_for_each(
        tf::make_sequence_range(lo, hi),
        [&](std::size_t i) {
          const auto &node = nodes[i];
          if (node.is_leaf() || node.is_empty())
            return;
          const auto data = node.get_data();
          auto &m = moments[i];
          double centroid_sum[3] = {0, 0, 0};
          double directed_area[3] = {0, 0, 0};
          double area = 0;
          for (auto c = std::size_t(data[0]), end = c + std::size_t(data[1]);
               c != end; ++c) {
            const auto &child = moments[c];
            for (std::size_t d = 0; d < 3; ++d) {
              centroid_sum[d] += child.area * child.position[d];
              directed_area[d] += child.directed_area[d];
            }
            area += child.area;
          }
          double centroid[3] = {0, 0, 0};
          if (area > 0)
            for (std::size_t d = 0; d < 3; ++d)
              centroid[d] = centroid_sum[d] / area;
          double second[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
          double third[18] = {0};
          const std::size_t pair_j[6] = {0, 0, 0, 1, 1, 2};
          const std::size_t pair_k[6] = {0, 1, 2, 1, 2, 2};
          for (auto c = std::size_t(data[0]), end = c + std::size_t(data[1]);
               c != end; ++c) {
            const auto &child = moments[c];
            double e[3];
            for (std::size_t d = 0; d < 3; ++d)
              e[d] = child.position[d] - centroid[d];
            for (std::size_t r = 0; r < 3; ++r) {
              for (std::size_t col = 0; col < 3; ++col)
                second[3 * r + col] += child.second_moment[3 * r + col] +
                                       child.directed_area[r] * e[col];
              for (std::size_t s = 0; s < 6; ++s) {
                const auto j = pair_j[s];
                const auto k = pair_k[s];
                third[6 * r + s] += child.third_moment[6 * r + s] +
                                    child.second_moment[3 * r + j] * e[k] +
                                    child.second_moment[3 * r + k] * e[j] +
                                    child.directed_area[r] * e[j] * e[k];
              }
            }
          }
          for (std::size_t d = 0; d < 3; ++d) {
            m.position[d] = centroid[d];
            m.directed_area[d] = directed_area[d];
          }
          for (std::size_t k = 0; k < 9; ++k)
            m.second_moment[k] = second[k];
          for (std::size_t k = 0; k < 18; ++k)
            m.third_moment[k] = third[k];
          m.area = area;
        },
        tf::checked);
  }
}

} // namespace tf::spatial
