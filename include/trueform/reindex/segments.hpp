/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/index_map.hpp"
#include "../core/segments.hpp"
#include "../core/segments_buffer.hpp"
#include "../core/views/block_indirect_range.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {
template <typename Policy, typename Range0, typename Range1, typename Range2,
          typename Range3, typename Policy1>
auto reindexed(const tf::segments<Policy> &segments,
               const tf::index_map<Range0, Range1> &edge_im,
               const tf::index_map<Range2, Range3> &point_im,
               tf::segments<Policy1> &out) {
  tf::parallel_copy(
      tf::make_indirect_range(
          edge_im.kept_ids(),
          tf::make_block_indirect_range(segments.edges(), point_im.f())),
      out.edges());
  tf::parallel_copy(
      tf::make_indirect_range(point_im.kept_ids(), segments.points()),
      out.points());
}

template <typename Policy, typename Range0, typename Range1, typename Range2,
          typename Range3, typename Index, typename RealT, std::size_t Dims>
auto reindexed(const tf::segments<Policy> &segments,
               const tf::index_map<Range0, Range1> &edge_im,
               const tf::index_map<Range2, Range3> &point_im,
               tf::segments_buffer<Index, RealT, Dims> &out) {
  out.edges_buffer().allocate(edge_im.kept_ids().size());
  out.points_buffer().allocate(point_im.kept_ids().size());
  auto out_s = out.segments();
  reindexed(segments, edge_im, point_im, out_s);
}

template <typename Policy, typename Range0, typename Range1, typename Range2,
          typename Range3>
auto reindexed(const tf::segments<Policy> &segments,
               const tf::index_map<Range0, Range1> &edge_im,
               const tf::index_map<Range2, Range3> &point_im) {
  tf::segments_buffer<std::decay_t<decltype(edge_im.kept_ids()[0])>,
                      tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>>
      out;
  reindexed(segments, edge_im, point_im, out);
  return out;
}
} // namespace tf
