/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../policy/unwrap.hpp"
#include "../polygon.hpp"
namespace tf::views {
template <typename Range0> struct polygons_dref {
  Range0 points;
  template <typename Range> auto operator()(Range &&ids) const {
    return tf::make_polygon(ids, points);
  }
};

template <typename Iterator0, typename Range1>
auto make_polygon_range_iter(Iterator0 faces_iter, Range1 &&points) {
  auto pts = tf::make_range(points);
  return iter::make_mapped(faces_iter, polygons_dref<decltype(pts)>{pts});
}

template <typename Range0, typename Range1> struct polygons {
  using iterator = decltype(tf::views::make_polygon_range_iter(
      std::declval<const Range0>().begin(),
      unwrapped(std::declval<const Range1>())));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using const_iterator = iterator;
  using size_type = std::size_t;

  polygons(const Range0 &faces, const Range1 &points)
      : _faces(faces), _points{points} {}

  auto faces() const -> const Range0 & { return _faces; }

  auto points() const -> const Range1 & { return _points; }

  auto begin() const {
    return views::make_polygon_range_iter(_faces.begin(), unwrapped(_points));
  }

  auto end() const {
    return views::make_polygon_range_iter(_faces.end(), unwrapped(_points));
  }

  auto size() const -> size_type { return _faces.size(); }
  auto empty() const -> bool { return _faces.size() == 0; }

  auto front() const -> reference { return *begin(); }
  auto back() const -> reference { return *(begin() + size() - 1); }
  auto operator[](size_type i) const -> reference { return *(begin() + i); }

private:
  Range0 _faces;
  Range1 _points;
};
} // namespace tf::views
