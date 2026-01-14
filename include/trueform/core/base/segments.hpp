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

#include "../iter/mapped_iterator.hpp"
#include "../policy/unwrap.hpp"
#include "../segment.hpp"

namespace tf::core {
template <typename Range0> struct segment_dref {
  Range0 points;
  template <typename Range> auto operator()(Range &&ids) const {
    return tf::make_segment(ids, points);
  }
};

template <typename Iterator0, typename Range1>
auto make_segment_range_iter(Iterator0 edges_iter, Range1 &&points) {
  auto pts = tf::make_range(points);
  return iter::make_mapped(edges_iter, segment_dref<decltype(pts)>{pts});
}

template <typename Range0, typename Range1> struct segments {
  using const_iterator = decltype(tf::core::make_segment_range_iter(
      std::declval<const Range0 &>().begin(),
      unwrapped(std::declval<const Range1 &>())));
  using iterator = decltype(tf::core::make_segment_range_iter(
      std::declval<Range0 &>().begin(), unwrapped(std::declval<Range1 &>())));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

  segments(const Range0 &edges, const Range1 &points)
      : _edges(edges), _points{points} {}

  segments(Range0 &&edges, Range1 &&points)
      : _edges(std::move(edges)), _points{std::move(points)} {}

  segments(const Range0 &edges, Range1 &&points)
      : _edges(edges), _points{std::move(points)} {}

  segments(Range0 &&edges, const Range1 &points)
      : _edges(std::move(edges)), _points{points} {}

  auto edges() const -> const Range0 & { return _edges; }

  auto edges() -> Range0 & { return _edges; }

  auto points() const -> const Range1 & { return _points; }

  auto points() -> Range1 & { return _points; }

  auto begin() const -> const_iterator {
    return core::make_segment_range_iter(_edges.begin(), unwrapped(_points));
  }

  auto begin() -> iterator {
    return core::make_segment_range_iter(_edges.begin(), unwrapped(_points));
  }

  auto end() const -> const_iterator {
    return core::make_segment_range_iter(_edges.end(), unwrapped(_points));
  }

  auto end() -> iterator {
    return core::make_segment_range_iter(_edges.end(), unwrapped(_points));
  }

  auto size() const -> size_type { return _edges.size(); }

  auto empty() const -> bool { return _edges.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(begin() + size() - 1); }

  auto back() -> reference { return *(begin() + size() - 1); }

  auto operator[](size_type i) const -> const_reference {
    return *(begin() + i);
  }

  auto operator[](size_type i) -> reference { return *(begin() + i); }

private:
  Range0 _edges;
  Range1 _points;
};

template <typename Range0, typename Range1>
auto make_segments(Range0 &&_edges, Range1 &&_points) {
  return segments<std::decay_t<Range0>, std::decay_t<Range1>>{
      static_cast<Range0 &&>(_edges), static_cast<Range1 &&>(_points)};
}

template <typename Range0, typename Range1>
auto has_edges_policy(const segments<Range0, Range1> *) -> std::true_type;
auto has_edges_policy(const void *) -> std::false_type;
} // namespace tf::core
namespace tf {
template <typename T>
inline constexpr bool has_edges = decltype(has_edges_policy(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;
}
