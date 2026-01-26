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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/index_map.hpp"
#include "../core/segments.hpp"
#include "../core/segments_buffer.hpp"
#include "../core/views/block_indirect_range.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {

/// @ingroup reindex
/// @brief Apply index maps to segments (output parameter).
///
/// Applies edge and point index maps to reindex segments.
/// Edge vertex indices are remapped according to point_im.
///
/// @tparam Policy The policy type of the input segments.
/// @tparam Range0 Edge index map kept IDs range type.
/// @tparam Range1 Edge index map function range type.
/// @tparam Range2 Point index map kept IDs range type.
/// @tparam Range3 Point index map function range type.
/// @tparam Policy1 Output segments policy type.
/// @param segments The input @ref tf::segments.
/// @param edge_im Edge @ref tf::index_map to apply.
/// @param point_im Point @ref tf::index_map to apply.
/// @param out Output @ref tf::segments to populate.
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

/// @ingroup reindex
/// @brief Apply index maps to segments (segments_buffer output parameter).
/// @overload
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

/// @ingroup reindex
/// @brief Apply index maps to segments.
///
/// @tparam Policy The policy type of the segments.
/// @tparam Range0 Edge index map kept IDs range type.
/// @tparam Range1 Edge index map function range type.
/// @tparam Range2 Point index map kept IDs range type.
/// @tparam Range3 Point index map function range type.
/// @param segments The input @ref tf::segments.
/// @param edge_im Edge @ref tf::index_map to apply.
/// @param point_im Point @ref tf::index_map to apply.
/// @return A @ref tf::segments_buffer with reindexed geometry.
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
