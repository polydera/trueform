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
#include "../../core/grain.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./delaunay_execution_tuning.hpp"
#include "./load_delaunay_sites.hpp"
#include <cstddef>
#include <tbb/task_arena.h>

namespace tf::topology::cdt {

template <typename Owner, typename Points, typename Converter>
auto load_delaunay_sites_parallel(Owner &owner, const Points &points,
                                  const Converter &converter) -> void {
  using Index = typename Owner::index_type;
  if (points.size() < delaunay_execution_tuning::parallel_load_sites ||
      tbb::this_task_arena::max_concurrency() < 2) {
    load_delaunay_sites(owner, points, converter);
    return;
  }
  owner._sites.allocate(points.size());
  tf::parallel_for_each(
      tf::make_sequence_range(points.size()),
      [&](std::size_t i) {
        const auto point = converter(points[i]);
        owner._sites[i] = {point[0], point[1], static_cast<Index>(i), Index(0)};
      },
      tf::grain(delaunay_execution_tuning::parallel_task_sites));
}

} // namespace tf::topology::cdt
