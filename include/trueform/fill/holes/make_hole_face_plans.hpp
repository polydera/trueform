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
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./subdivide_carrier_triangle.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::fill {

/// The plan of every carrying face the surviving splits reached.
///
/// The splits arrive in the canonical order the result publishes them in —
/// face, then edge key, then parameter — so a face's run is contiguous and
/// already in the order its subdivision consumes. Block `i` of the product
/// is the plan of `plan_faces[i]`.
template <typename Policy, typename Iterator, typename Index>
auto make_hole_face_plans(
    const tf::polygons<Policy> &polygons,
    const tf::range<Iterator, tf::dynamic_size> &splits,
    tf::buffer<Index> &plan_faces,
    tf::offset_block_buffer<Index, std::array<Index, 3>> &plan_triangles)
    -> void {
  plan_faces.clear();
  plan_triangles.clear();
  if (!splits.size())
    return;

  tf::buffer<Index> starts;
  for (std::size_t k = 0; k < splits.size(); ++k)
    if (!k || splits[k].face != splits[k - 1].face) {
      plan_faces.push_back(splits[k].face);
      starts.push_back(Index(k));
    }
  starts.push_back(Index(splits.size()));

  struct local_t {
    tf::fill::hole_plan_scratch<Index> scratch;
    tf::buffer<Index> sizes;
    tf::buffer<std::array<Index, 3>> data;
  };
  const auto carriers = polygons.faces();
  auto &offsets = plan_triangles.offsets_buffer();
  auto &data = plan_triangles.data_buffer();
  offsets.push_back(0);

  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(plan_faces.size()),
      std::tie(offsets, data), local_t{},
      [&carriers, &splits, &starts](auto block, local_t &local) {
        local.sizes.clear();
        local.data.clear();
        for (auto run : block) {
          const auto face = carriers[std::size_t(
              splits[std::size_t(starts[std::size_t(run)])].face)];
          const std::array<Index, 3> corners{face[0], face[1], face[2]};
          const auto before = local.data.size();
          tf::fill::subdivide_carrier_triangle(
              corners,
              tf::make_range(splits.begin() + starts[std::size_t(run)],
                             splits.begin() + starts[std::size_t(run) + 1]),
              local.scratch, local.data);
          local.sizes.push_back(Index(local.data.size() - before));
        }
      },
      [](const local_t &local, auto &result) {
        auto &[offsets, data] = result;
        for (auto size : local.sizes)
          offsets.push_back(offsets.back() + size);
        const auto base = data.size();
        data.reallocate(base + local.data.size());
        std::copy(local.data.begin(), local.data.end(),
                  data.begin() + std::ptrdiff_t(base));
      });
}

} // namespace tf::fill
