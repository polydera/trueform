/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./mapped_range.hpp"
#include "./point_range.hpp"
#include "./segment.hpp"

namespace tf {
namespace implementation {
template <typename Range0> struct segment_range_policy {
  Range0 points;
  template <typename Range> auto operator()(Range &&ids) const {
    return tf::make_segment(ids, points);
  }
};
} // namespace implementation

template <typename Range0, typename Range1, typename...>
struct segment_range
    : decltype(tf::make_mapped_range(
          std::declval<Range0>(),
          std::declval<implementation::segment_range_policy<Range1>>())) {
private:
  using base_t = decltype(tf::make_mapped_range(
      std::declval<Range0>(),
      std::declval<implementation::segment_range_policy<Range1>>()));

public:
  segment_range(const Range0 &faces, const Range1 &points)
      : base_t{tf::make_mapped_range(
            faces, implementation::segment_range_policy<Range1>{points})} {}

  auto edges() const {
    return tf::make_range(base_t::begin().base_iter(), base_t::size());
  }

  auto points() const {
    return tf::make_point_range(base_t::begin().dereference_policy().points);
  }
};

template <typename Range0, typename Range1>
auto make_segment_range(Range0 &&edges, Range1 &&points) {
  auto r0 = tf::make_range(edges);
  auto r1 = tf::make_point_range(points);
  return segment_range<decltype(r0), decltype(r1)>{r0, r1};
}
} // namespace tf
