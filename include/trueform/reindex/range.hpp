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
#include "../core/views/indirect_range.hpp"
namespace tf {

/// @ingroup reindex
/// @brief Apply index map to a generic range (output parameter).
///
/// Gathers elements from range_in at positions specified by im.kept_ids()
/// and writes them to range_out.
///
/// @tparam Iter0 Input iterator type.
/// @tparam N0 Static size hint.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @tparam Iter1 Output iterator type.
/// @tparam N1 Output static size hint.
/// @param range_in The input @ref tf::range.
/// @param im The @ref tf::index_map to apply.
/// @param range_out Output range to populate.
template <typename Iter0, std::size_t N0, typename Range0, typename Range1,
          typename Iter1, std::size_t N1>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im,
               tf::range<Iter1, N1> range_out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), range_in),
                    range_out);
}

/// @ingroup reindex
/// @brief Apply index map to a range (std::vector output parameter).
/// @overload
template <typename Iter0, std::size_t N0, typename Range0, typename Range1,
          typename T>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im, std::vector<T> &out) {
  out.resize(im.kept_ids().size());
  auto r = tf::make_range(out);
  reindexed(range_in, im, r);
}

/// @ingroup reindex
/// @brief Apply index map to a range (tf::buffer output parameter).
/// @overload
template <typename Iter0, std::size_t N0, typename Range0, typename Range1,
          typename T>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im, tf::buffer<T> &out) {
  out.allocate(im.kept_ids().size());
  auto r = tf::make_range(out);
  reindexed(range_in, im, r);
}

/// @ingroup reindex
/// @brief Apply index map to a range.
///
/// Returns a @ref tf::buffer or std::vector depending on element type.
///
/// @tparam Iter0 Input iterator type.
/// @tparam N0 Static size hint.
/// @tparam Range0 Index map kept IDs range type.
/// @tparam Range1 Index map function range type.
/// @param range_in The input @ref tf::range.
/// @param im The @ref tf::index_map to apply.
/// @return Reindexed buffer or vector.
template <typename Iter0, std::size_t N0, typename Range0, typename Range1>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im) {
  using T = typename std::iterator_traits<Iter0>::value_type;
  if constexpr (std::is_trivially_default_constructible<T>::value &&
                std::is_trivially_destructible<T>::value) {
    tf::buffer<T> out;
    reindexed(range_in, im, out);
    return out;
  } else {
    std::vector<T> out;
    reindexed(range_in, im, out);
    return out;
  }
}
} // namespace tf
