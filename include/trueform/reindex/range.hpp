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
template <typename Iter0, std::size_t N0, typename Range0, typename Range1,
          typename Iter1, std::size_t N1>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im,
               tf::range<Iter1, N1> range_out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), range_in),
                    range_out);
}

template <typename Iter0, std::size_t N0, typename Range0, typename Range1,
          typename T>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im, std::vector<T> &out) {
  out.resize(im.kept_ids().size());
  auto r = tf::make_range(out);
  reindexed(range_in, im, r);
}

template <typename Iter0, std::size_t N0, typename Range0, typename Range1,
          typename T>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im, tf::buffer<T> &out) {
  out.allocate(im.kept_ids().size());
  auto r = tf::make_range(out);
  reindexed(range_in, im, r);
}

template <typename Iter0, std::size_t N0, typename Range0, typename Range1>
auto reindexed(const tf::range<Iter0, N0> &range_in,
               const tf::index_map<Range0, Range1> &im) {
  using T = typename Iter0::value_type;
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
