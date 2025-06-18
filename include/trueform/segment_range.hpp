/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./implementation/mapped_iterator.hpp"
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

template <typename Range0, typename Range1, typename...> struct segment_range {
  using iterator = decltype(tf::implementation::iter::make_mapped(
      std::declval<const Range0>().begin(),
      implementation::segment_range_policy<Range1>{
          tf::make_range(std::declval<const Range1>())}));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using const_iterator = iterator;
  using size_type = std::size_t;

  segment_range(const Range0 &edges, const Range1 &points)
      : _edges(edges), _points{points} {}

  auto edges() const -> const Range0 & { return _edges; }

  auto points() const -> const Range1 & { return _points; }

  auto begin() const {
    return tf::implementation::iter::make_mapped(
        _edges.begin(),
        implementation::segment_range_policy<Range1>{tf::make_range(_points)});
  }

  auto end() const {
    return tf::implementation::iter::make_mapped(
        _edges.end(),
        implementation::segment_range_policy<Range1>{tf::make_range(_points)});
  }

  auto size() const -> size_type { return _edges.size(); }
  auto empty() const -> bool { return _edges.size() == 0; }

  auto front() const -> reference { return *begin(); }
  auto back() const -> reference { return *(begin() + size() - 1); }
  auto operator[](size_type i) const -> reference { return *(begin() + i); }

private:
  Range0 _edges;
  Range1 _points;
};
} // namespace tf
