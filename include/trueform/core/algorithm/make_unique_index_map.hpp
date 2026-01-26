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
#include "../index_map.hpp"
#include "../views/zip.hpp"
#include "./parallel_iota.hpp"
#include "tbb/parallel_sort.h"

namespace tf {

/// @ingroup core_algorithms
/// @brief Create index map for unique elements.
///
/// Builds a mapping where duplicate elements map to the same ID.
/// Uses parallel sort for efficiency on large data.
///
/// @tparam Range The input data range type.
/// @tparam Index The index type.
/// @tparam CompareEq Equality comparison predicate.
/// @tparam CompareLess Less-than comparison predicate.
/// @param data Input data to deduplicate.
/// @param im Output index map buffer.
/// @param eq_f Equality predicate.
/// @param less_f Ordering predicate.
template <typename Range, typename Index, typename CompareEq,
          typename CompareLess>
auto make_unique_index_map(const Range &data, tf::index_map_buffer<Index> &im,
                           CompareEq eq_f, CompareLess less_f) {
  if (!data.size())
    return;
  im.kept_ids().allocate(data.size());
  tf::parallel_iota(im.kept_ids(), 0);
  tbb::parallel_sort(im.kept_ids(), [&](const auto &x0, const auto &x1) {
    return less_f(data[x0], data[x1]);
  });

  im.f().allocate(data.size());

  auto iter = im.kept_ids().begin();
  auto end = im.kept_ids().end();
  Index current_id = 0;
  im.f()[*iter] = current_id;
  auto check_with = iter;
  auto write_to = iter + 1;
  while (++iter != end) {
    if (!eq_f(data[*check_with], data[*iter])) {
      ++current_id;
      check_with = iter;
      *write_to++ = *iter;
    }
    im.f()[*iter] = current_id;
  }
  im.kept_ids().erase(im.kept_ids().begin() + current_id + 1,
                      im.kept_ids().end());
}

/// @ingroup core_algorithms
/// @brief Create index map and compact data to unique elements.
///
/// Similar to make_unique_index_map but also removes duplicates
/// from the input data range in-place.
///
/// @tparam Range The input data range type.
/// @tparam Index The index type.
/// @tparam CompareEq Equality comparison predicate.
/// @tparam CompareLess Less-than comparison predicate.
/// @param data Input data to deduplicate (modified in-place).
/// @param im Output index map buffer.
/// @param eq_f Equality predicate.
/// @param less_f Ordering predicate.
/// @return Iterator to the new end of unique data.
template <typename Range, typename Index, typename CompareEq,
          typename CompareLess>
auto make_unique_and_index_map(Range &data, tf::index_map_buffer<Index> &im,
                               CompareEq eq_f, CompareLess less_f) {
  if (!data.size())
    return data.end();
  im.kept_ids().allocate(data.size());
  tf::parallel_iota(im.kept_ids(), 0);
  auto zipped_data = tf::zip(data, im.kept_ids());
  tbb::parallel_sort(zipped_data.begin(), zipped_data.end(),
                     [&](const auto &x0, const auto &x1) {
                       using std::get;
                       return less_f(get<0>(x0), get<0>(x1));
                     });

  im.f().allocate(data.size());

  auto iter = im.kept_ids().begin();
  auto end = im.kept_ids().end();
  //
  auto data_iter = data.begin();
  //
  Index current_id = 0;
  im.f()[*iter] = current_id;
  auto check_with = data_iter;
  auto write_to_index = iter + 1;
  auto write_to_data = data_iter + 1;
  while (++iter != end) {
    ++data_iter;
    if (!eq_f(*check_with, *data_iter)) {
      ++current_id;
      check_with = data_iter;
      *write_to_index++ = *iter;
      *write_to_data++ = *data_iter;
    }
    im.f()[*iter] = current_id;
  }
  im.kept_ids().erase(im.kept_ids().begin() + current_id + 1,
                      im.kept_ids().end());
  return write_to_data;
}

/// @ingroup core_algorithms
/// @brief Create index map for unique elements with default comparators.
/// @overload
template <typename Range, typename Index>
auto make_unique_index_map(const Range &data, tf::index_map_buffer<Index> &im) {
  return make_unique_index_map(data, im, std::equal_to{}, std::less<>{});
}

/// @ingroup core_algorithms
/// @brief Create index map and compact data with default comparators.
/// @overload
template <typename Range, typename Index>
auto make_unique_and_index_map(Range &data, tf::index_map_buffer<Index> &im) {
  return make_unique_and_index_map(data, im, std::equal_to{}, std::less<>{});
}
} // namespace tf
