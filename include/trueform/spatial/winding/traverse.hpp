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
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../tree/winding_block.hpp"
#include "./solid_angle.hpp"
#include "./winding_tree.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace tf::spatial {

/// The winding accumulation over one moments-capable tree, one child
/// block per visit: the far test and the far-field polynomial — the
/// folded combination scheme the owning header cites, on the
/// normalized direction — run branchless over the four
/// lanes, far lanes answer in place, near leaves contribute exact
/// solid angles, and only near inner children re-enter the stack. A
/// zero-area lane states its own irrelevance. Returns the unnormalized
/// sum — the winding number is the sum over 4π. The stack is
/// caller-owned scratch, cleared here, so a batch reuses its capacity
/// across queries.
template <typename Winding, typename Polygons, typename Index>
auto winding_traverse(const Winding &winding, const Polygons &polygons,
                      const tf::point<double, 3> &qp, double beta2,
                      tf::small_vector<Index, 512> &stack) -> double {
  const auto &nodes = winding.nodes;
  const auto &ids = winding.ids;
  const auto &blocks = winding.moments.blocks;
  const auto &tickets = winding.moments.tickets;
  if (!nodes.size())
    return 0;
  double sum = 0;
  const auto leaf_contribution = [&](const auto data) {
    double part = 0;
    for (auto it = ids.begin() + data[0], end = it + data[1]; it != end; ++it) {
      const auto polygon = polygons[*it];
      const auto sz = std::size_t(polygon.size());
      const auto p0 = polygon[0].template as<double>();
      for (std::size_t j = 1; j + 1 < sz; ++j)
        part += triangle_solid_angle(qp, p0, polygon[j].template as<double>(),
                                     polygon[j + 1].template as<double>());
    }
    return part;
  };
  if (nodes[0].is_leaf())
    return leaf_contribution(nodes[0].get_data());
  using real_t = std::decay_t<decltype(blocks.begin()->area[0])>;
  const auto beta2_r = real_t(beta2);
  const real_t qr[3] = {real_t(qp[0]), real_t(qp[1]), real_t(qp[2])};
  const auto push_blocks = [&](std::size_t node_id) {
    const auto first = tickets[std::size_t(node_id)];
    const auto n_sub =
        sub_block_count(std::size_t(nodes[node_id].get_data()[1]));
    for (std::size_t s = 0; s < n_sub; ++s)
      stack.push_back(Index(std::size_t(first) + s));
  };
  assert(tickets[0] >= 0 && "a non-leaf root owns a child block");
  stack.clear();
  push_blocks(0);
  while (stack.size()) {
    const auto &block = blocks[std::size_t(stack.back())];
    stack.pop_back();
    real_t q[3][4];
    real_t dd[4];
    for (std::size_t l = 0; l < 4; ++l) {
      real_t acc = 0;
      for (std::size_t d = 0; d < 3; ++d) {
        q[d][l] = qr[d] - block.position[d][l];
        acc += q[d][l] * q[d][l];
      }
      dd[l] = acc;
    }
    bool is_far[4];
    for (std::size_t l = 0; l < 4; ++l)
      is_far[l] = unsigned(dd[l] > beta2_r * block.far_field_radius2[l]) &
                  unsigned(dd[l] > real_t(0)) &
                  unsigned(block.area[l] > real_t(0));
    real_t omega[4];
    for (std::size_t l = 0; l < 4; ++l) {
      const auto safe = is_far[l] ? dd[l] : real_t(1);
      const auto qm2 = real_t(1) / safe;
      const auto qm1 = std::sqrt(qm2);
      const real_t qh[3] = {q[0][l] * qm1, q[1][l] * qm1, q[2][l] * qm1};
      const real_t q2[3] = {qh[0] * qh[0], qh[1] * qh[1], qh[2] * qh[2]};
      real_t o = -(qh[0] * block.directed_area[0][l] +
                   qh[1] * block.directed_area[1][l] +
                   qh[2] * block.directed_area[2][l]) *
                 qm2;
      const auto qm3 = qm2 * qm1;
      const auto d0 = block.second_diag[0][l];
      const auto d1 = block.second_diag[1][l];
      const auto d2 = block.second_diag[2][l];
      o += qm3 * (d0 + d1 + d2 -
                  real_t(3) * (q2[0] * d0 + q2[1] * d1 + q2[2] * d2 +
                               qh[0] * qh[1] * block.second_cross[0][l] +
                               qh[0] * qh[2] * block.second_cross[1][l] +
                               qh[1] * qh[2] * block.second_cross[2][l]));
      const auto qm4 = qm2 * qm2;
      const real_t x0 = block.third_mixed[0][l];
      const real_t x1 = block.third_mixed[1][l];
      const real_t x2 = block.third_mixed[2][l];
      const real_t x3 = block.third_mixed[3][l];
      const real_t x4 = block.third_mixed[4][l];
      const real_t x5 = block.third_mixed[5][l];
      const real_t t0[3] = {x3 + x4, x5 + x0, x1 + x2};
      const real_t t1[3] = {qh[1] * x0 + qh[2] * x1, qh[2] * x2 + qh[0] * x3,
                            qh[0] * x4 + qh[1] * x5};
      real_t lin = 0;
      real_t cubic = 0;
      real_t quad = 0;
      for (std::size_t d = 0; d < 3; ++d) {
        const auto diag = block.third_diag[d][l];
        lin += qh[d] * (real_t(3) * diag + t0[d]);
        cubic += q2[d] * qh[d] * diag;
        quad += q2[d] * t1[d];
      }
      o +=
          qm4 *
          (real_t(1.5) * lin -
           real_t(7.5) *
               (cubic + qh[0] * qh[1] * qh[2] * block.third_permute[l] + quad));
      omega[l] = is_far[l] ? o : real_t(0);
    }
    sum += double(omega[0]) + double(omega[1]) + double(omega[2]) +
           double(omega[3]);
    for (std::size_t l = 0; l < std::size_t(block.n_children); ++l) {
      if (is_far[l] || !(block.area[l] > real_t(0)))
        continue;
      const auto child = std::size_t(block.first_child) + l;
      const auto &node = nodes[child];
      if (node.is_leaf()) {
        sum += leaf_contribution(node.get_data());
        continue;
      }
      push_blocks(child);
    }
  }
  return sum;
}

} // namespace tf::spatial
