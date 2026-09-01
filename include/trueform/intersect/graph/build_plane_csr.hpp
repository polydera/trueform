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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/enumerate.hpp"
#include "tbb/parallel_sort.h"
#include <cstddef>
#include <iterator>
#include <tuple>

namespace tf::intersect::graph {

/// plane -> members: sort compact {plane, group} records by value, the
/// offsets are the key boundaries. Every plane owns the group that
/// named it, so the boundaries are exactly one block per plane.
template <typename Index>
auto build_plane_csr(const tf::buffer<Index> &plane_of, Index n_planes,
                     tf::buffer<Index> &plane_offsets,
                     tf::buffer<Index> &plane_data) -> void {
  const auto n_groups = plane_of.size();
  struct prec_t {
    Index plane, group;
  };
  tf::buffer<prec_t> precs;
  precs.allocate(n_groups);
  tf::parallel_for_each(
      tf::enumerate(plane_of),
      [&](auto pair) {
        auto &&[g, p] = pair;
        precs[std::size_t(g)] = {p, Index(g)};
      },
      tf::checked);
  tbb::parallel_sort(precs.begin(), precs.end(),
                     [](const prec_t &x, const prec_t &y) {
                       return std::tie(x.plane, x.group) <
                              std::tie(y.plane, y.group);
                     });
  plane_offsets.clear();
  plane_offsets.reserve(std::size_t(n_planes) + 1);
  tf::compute_offsets(
      precs, std::back_inserter(plane_offsets), Index(0),
      [](const prec_t &x, const prec_t &y) { return x.plane == y.plane; });
  plane_data.allocate(n_groups);
  tf::parallel_for_each(
      tf::enumerate(precs),
      [&](auto pair) {
        auto &&[i, rec] = pair;
        plane_data[std::size_t(i)] = rec.group;
      },
      tf::checked);
}

} // namespace tf::intersect::graph
