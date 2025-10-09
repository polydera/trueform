/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/index_map.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {
template <typename Policy, typename Range0, typename Range1, typename Policy1>
auto reindexed(const tf::points<Policy> &points,
               const tf::index_map<Range0, Range1> &im,
               tf::points<Policy1> &out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), points), out);
}

template <typename Policy, typename Range0, typename Range1, typename RealT,
          std::size_t Dims>
auto reindexed(const tf::points<Policy> &points,
               const tf::index_map<Range0, Range1> &im,
               tf::points_buffer<RealT, Dims> &out) {
  out.allocate(im.kept_ids().size());
  auto out_p = out.points();
  reindexed(points, im, out_p);
}

template <typename Policy, typename Range0, typename Range1>
auto reindexed(const tf::points<Policy> &points,
               const tf::index_map<Range0, Range1> &im) {
  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      out;
  reindexed(points, im, out);
  return out;
}
} // namespace tf
