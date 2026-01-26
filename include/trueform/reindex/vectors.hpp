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
#include "../core/vectors.hpp"
#include "../core/vectors_buffer.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {

/// @ingroup reindex
/// @brief Apply index map to vectors (output parameter).
///
/// Gathers vectors at positions specified by im.kept_ids().
///
/// @tparam Policy The policy type of the input vectors.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @tparam Policy1 Output vectors policy type.
/// @param vectors The input @ref tf::vectors.
/// @param im The @ref tf::index_map to apply.
/// @param out Output @ref tf::vectors to populate.
template <typename Policy, typename Range0, typename Range1, typename Policy1>
auto reindexed(const tf::vectors<Policy> &vectors,
               const tf::index_map<Range0, Range1> &im,
               tf::vectors<Policy1> &out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), vectors), out);
}

/// @ingroup reindex
/// @brief Apply index map to vectors (vectors_buffer output parameter).
/// @overload
template <typename Policy, typename Range0, typename Range1, typename RealT,
          std::size_t Dims>
auto reindexed(const tf::vectors<Policy> &vectors,
               const tf::index_map<Range0, Range1> &im,
               tf::vectors_buffer<RealT, Dims> &out) {
  out.allocate(im.kept_ids().size());
  auto out_p = out.vectors();
  reindexed(vectors, im, out_p);
}

/// @ingroup reindex
/// @brief Apply index map to vectors.
///
/// @tparam Policy The policy type of the vectors.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @param vectors The input @ref tf::vectors.
/// @param im The @ref tf::index_map to apply.
/// @return A @ref tf::vectors_buffer with reindexed vectors.
template <typename Policy, typename Range0, typename Range1>
auto reindexed(const tf::vectors<Policy> &vectors,
               const tf::index_map<Range0, Range1> &im) {
  tf::vectors_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      out;
  reindexed(vectors, im, out);
  return out;
}
} // namespace tf

