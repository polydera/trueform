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
#include "../core/coordinate_dims.hpp"
#include "../core/index_map.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {

/// @ingroup reindex
/// @brief Apply index map to points (output parameter).
///
/// Gathers points at positions specified by im.kept_ids().
///
/// @tparam Policy The policy type of the input points.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @tparam Policy1 Output points policy type.
/// @param points The input @ref tf::points.
/// @param im The @ref tf::index_map to apply.
/// @param out Output @ref tf::points to populate.
template <typename Policy, typename Range0, typename Range1, typename Policy1>
auto reindexed(const tf::points<Policy> &points,
               const tf::index_map<Range0, Range1> &im,
               tf::points<Policy1> &out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), points), out);
}

/// @ingroup reindex
/// @brief Apply index map to points (points_buffer output parameter).
/// @overload
template <typename Policy, typename Range0, typename Range1, typename RealT,
          std::size_t Dims>
auto reindexed(const tf::points<Policy> &points,
               const tf::index_map<Range0, Range1> &im,
               tf::points_buffer<RealT, Dims> &out) {
  out.allocate(im.kept_ids().size());
  auto out_p = out.points();
  reindexed(points, im, out_p);
}

/// @ingroup reindex
/// @brief Apply index map to points.
///
/// @tparam Policy The policy type of the points.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @param points The input @ref tf::points.
/// @param im The @ref tf::index_map to apply.
/// @return A @ref tf::points_buffer with reindexed points.
template <typename Policy, typename Range0, typename Range1>
auto reindexed(const tf::points<Policy> &points,
               const tf::index_map<Range0, Range1> &im) {
  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      out;
  reindexed(points, im, out);
  return out;
}
} // namespace tf
