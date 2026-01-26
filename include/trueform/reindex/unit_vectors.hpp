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
#include "../core/unit_vectors.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {

/// @ingroup reindex
/// @brief Apply index map to unit vectors (output parameter).
///
/// Gathers unit vectors at positions specified by im.kept_ids().
///
/// @tparam Policy The policy type of the input unit vectors.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @tparam Policy1 Output unit vectors policy type.
/// @param unit_vectors The input @ref tf::unit_vectors.
/// @param im The @ref tf::index_map to apply.
/// @param out Output @ref tf::unit_vectors to populate.
template <typename Policy, typename Range0, typename Range1, typename Policy1>
auto reindexed(const tf::unit_vectors<Policy> &unit_vectors,
               const tf::index_map<Range0, Range1> &im,
               tf::unit_vectors<Policy1> &out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), unit_vectors), out);
}

/// @ingroup reindex
/// @brief Apply index map to unit vectors (unit_vectors_buffer output parameter).
/// @overload
template <typename Policy, typename Range0, typename Range1, typename RealT,
          std::size_t Dims>
auto reindexed(const tf::unit_vectors<Policy> &unit_vectors,
               const tf::index_map<Range0, Range1> &im,
               tf::unit_vectors_buffer<RealT, Dims> &out) {
  out.allocate(im.kept_ids().size());
  auto out_p = out.unit_vectors();
  reindexed(unit_vectors, im, out_p);
}

/// @ingroup reindex
/// @brief Apply index map to unit vectors.
///
/// @tparam Policy The policy type of the unit vectors.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @param unit_vectors The input @ref tf::unit_vectors.
/// @param im The @ref tf::index_map to apply.
/// @return A @ref tf::unit_vectors_buffer with reindexed unit vectors.
template <typename Policy, typename Range0, typename Range1>
auto reindexed(const tf::unit_vectors<Policy> &unit_vectors,
               const tf::index_map<Range0, Range1> &im) {
  tf::unit_vectors_buffer<tf::coordinate_type<Policy>,
                          tf::coordinate_dims_v<Policy>>
      out;
  reindexed(unit_vectors, im, out);
  return out;
}
} // namespace tf
