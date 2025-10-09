/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/index_map.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {
template <typename RangeIn, typename Range0, typename Range1, typename RangeOut>
auto reindexed_range(const RangeIn &range_in,
                     const tf::index_map<Range0, Range1> &im,
                     RangeOut &range_out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), range_in),
                    range_out);
}

template <typename RangeIn, typename Range0, typename Range1, typename T>
auto reindexed_range(const RangeIn &range_in,
                     const tf::index_map<Range0, Range1> &im,
                     std::vector<T> &out) {
  out.resize(im.kept_ids().size());
  auto r = tf::make_range(out);
  reindexed_range(range_in, im, out);
}

template <typename RangeIn, typename Range0, typename Range1, typename T>
auto reindexed_range(const RangeIn &range_in,
                     const tf::index_map<Range0, Range1> &im,
                     tf::buffer<T> &out) {
  out.allocate(im.kept_ids().size());
  auto r = tf::make_range(out);
  reindexed_range(range_in, im, out);
}

template <typename RangeIn, typename Range0, typename Range1, typename T>
auto reindexed_range(const RangeIn &range_in,
                     const tf::index_map<Range0, Range1> &im) {
  if constexpr (std::is_trivially_default_constructible<T>::value &&
                std::is_trivially_destructible<T>::value) {
    tf::buffer<typename RangeIn::value_type> out;
    reindexed_range(range_in, im, out);
    return out;
  } else {
    std::vector<typename RangeIn::value_type> out;
    reindexed_range(range_in, im, out);
    return out;
  }
}
} // namespace tf
