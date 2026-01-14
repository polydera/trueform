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

#include "../policy/unwrap.hpp"
#include "../polygon.hpp"
namespace tf::core {
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
  using const_iterator = decltype(tf::core::make_polygon_range_iter(
      std::declval<const Range0 &>().begin(),
      unwrapped(std::declval<const Range1 &>())));
  using iterator = decltype(tf::core::make_polygon_range_iter(
      std::declval<Range0 &>().begin(), unwrapped(std::declval<Range1 &>())));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

  polygons(const Range0 &faces, const Range1 &points)
      : _faces(faces), _points{points} {}

  polygons(Range0 &&faces, Range1 &&points)
      : _faces(std::move(faces)), _points{std::move(points)} {}

  polygons(const Range0 &faces, Range1 &&points)
      : _faces(faces), _points{std::move(points)} {}

  polygons(Range0 &&faces, const Range1 &points)
      : _faces(std::move(faces)), _points{points} {}

  auto faces() const -> const Range0 & { return _faces; }

  auto faces() -> Range0 & { return _faces; }

  auto points() const -> const Range1 & { return _points; }

  auto points() -> Range1 & { return _points; }

  auto begin() const -> const_iterator {
    return core::make_polygon_range_iter(_faces.begin(), unwrapped(_points));
  }

  auto begin() -> iterator {
    return core::make_polygon_range_iter(_faces.begin(), unwrapped(_points));
  }

  auto end() const -> const_iterator {
    return core::make_polygon_range_iter(_faces.end(), unwrapped(_points));
  }

  auto end() -> iterator {
    return core::make_polygon_range_iter(_faces.end(), unwrapped(_points));
  }

  auto size() const -> size_type { return _faces.size(); }

  auto empty() const -> bool { return _faces.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(begin() + size() - 1); }

  auto back() -> reference { return *(begin() + size() - 1); }

  auto operator[](size_type i) const -> const_reference {
    return *(begin() + i);
  }

  auto operator[](size_type i) -> reference { return *(begin() + i); }

private:
  Range0 _faces;
  Range1 _points;
};

template <typename Range0, typename Range1>
auto make_polygons(Range0 &&_faces, Range1 &&_points) {
  return polygons<std::decay_t<Range0>, std::decay_t<Range1>>{
      static_cast<Range0 &&>(_faces), static_cast<Range1 &&>(_points)};
}
} // namespace tf::core
