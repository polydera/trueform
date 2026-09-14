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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../tree/winding_block.hpp"
#include "./raw_winding_moment.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::spatial {

/// Fold the raw rows into the child blocks: one serial pass tickets
/// the inner nodes — an inner node owns one sub-block per four
/// children, consecutively — one parallel pass writes each block's
/// lanes: per child the evaluation's combinations and the far-field
/// radius against the child's own box. The symmetric index order is
/// xx, xy, xz, yy, yz, zz.
template <typename RealT, typename Nodes>
auto finalize_winding_moments(tf::buffer<winding_block<RealT>> &blocks,
                              tf::buffer<std::int32_t> &tickets,
                              const raw_winding_moment *raw, const Nodes &nodes)
    -> void {
  const auto n = nodes.size();
  assert(n <= std::size_t(std::numeric_limits<std::int32_t>::max()) &&
         "block tickets are int32");
  tickets.allocate(n);
  std::int32_t n_blocks = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto &node = nodes[i];
    if (node.is_leaf() || node.is_empty()) {
      tickets[i] = std::int32_t(-1);
      continue;
    }
    tickets[i] = n_blocks;
    n_blocks += std::int32_t(sub_block_count(std::size_t(node.get_data()[1])));
  }
  blocks.allocate(std::size_t(n_blocks));
  auto *out = blocks.begin();
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t{0}, n),
      [&](std::size_t i) {
        const auto ticket = tickets[i];
        if (ticket < 0)
          return;
        const auto data = nodes[i].get_data();
        const auto n_sub = sub_block_count(std::size_t(data[1]));
        for (std::size_t sub = 0; sub < n_sub; ++sub) {
          auto &block = out[std::size_t(ticket) + sub];
          block = winding_block<RealT>{};
          block.first_child = std::int32_t(std::size_t(data[0]) + 4 * sub);
          block.n_children = std::int32_t(
              std::min(std::size_t(4), std::size_t(data[1]) - 4 * sub));
          for (std::size_t lane = 0; lane < std::size_t(block.n_children);
               ++lane) {
            const auto child = std::size_t(block.first_child) + lane;
            if (nodes[child].is_empty())
              continue;
            const auto &r = raw[child];
            for (std::size_t d = 0; d < 3; ++d) {
              block.position[d][lane] = RealT(r.position[d]);
              block.directed_area[d][lane] = RealT(r.directed_area[d]);
            }
            block.area[lane] = RealT(r.area);
            double rr = 0;
            for (std::size_t d = 0; d < 3; ++d) {
              const auto &bv = nodes[child].bv;
              const auto to_min = r.position[d] - double(bv.min[d]);
              const auto to_max = r.position[d] - double(bv.max[d]);
              rr += std::max(to_min * to_min, to_max * to_max);
            }
            block.far_field_radius2[lane] = RealT(rr);
            const auto &M = r.second_moment;
            block.second_diag[0][lane] = RealT(M[0]);
            block.second_diag[1][lane] = RealT(M[4]);
            block.second_diag[2][lane] = RealT(M[8]);
            block.second_cross[0][lane] = RealT(M[1] + M[3]);
            block.second_cross[1][lane] = RealT(M[6] + M[2]);
            block.second_cross[2][lane] = RealT(M[5] + M[7]);
            const auto *t0 = r.third_moment.data();
            const auto *t1 = t0 + 6;
            const auto *t2 = t0 + 12;
            block.third_diag[0][lane] = RealT(t0[0]);
            block.third_diag[1][lane] = RealT(t1[3]);
            block.third_diag[2][lane] = RealT(t2[5]);
            block.third_permute[lane] = RealT(2. * (t0[4] + t1[2] + t2[1]));
            block.third_mixed[0][lane] = RealT(2. * t0[1] + t1[0]);
            block.third_mixed[1][lane] = RealT(2. * t0[2] + t2[0]);
            block.third_mixed[2][lane] = RealT(2. * t1[4] + t2[3]);
            block.third_mixed[3][lane] = RealT(2. * t1[1] + t0[3]);
            block.third_mixed[4][lane] = RealT(2. * t2[2] + t0[5]);
            block.third_mixed[5][lane] = RealT(2. * t2[4] + t1[5]);
          }
        }
      },
      tf::checked);
}

} // namespace tf::spatial
