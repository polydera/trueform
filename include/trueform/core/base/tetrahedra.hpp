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
#include "../range.hpp"
#include "../tetrahedron.hpp"
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace tf::core {
template <typename Range0> struct tetrahedron_dref {
  Range0 points;
  template <typename Range> auto operator()(Range &&ids) const {
    return tf::make_tetrahedron(ids, points);
  }
};

template <typename Iterator0, typename Range1>
auto make_tetrahedron_range_iter(Iterator0 tetras_iter, Range1 &&points) {
  auto pts = tf::make_range(points);
  return iter::make_mapped(tetras_iter, tetrahedron_dref<decltype(pts)>{pts});
}

template <typename Range0, typename Range1> struct tetrahedra {
  using const_iterator = decltype(tf::core::make_tetrahedron_range_iter(
      std::declval<const Range0 &>().begin(),
      unwrapped(std::declval<const Range1 &>())));
  using iterator = decltype(tf::core::make_tetrahedron_range_iter(
      std::declval<Range0 &>().begin(), unwrapped(std::declval<Range1 &>())));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

  tetrahedra(const Range0 &tetras, const Range1 &points)
      : _tetras(tetras), _points{points} {}

  tetrahedra(Range0 &&tetras, Range1 &&points)
      : _tetras(std::move(tetras)), _points{std::move(points)} {}

  tetrahedra(const Range0 &tetras, Range1 &&points)
      : _tetras(tetras), _points{std::move(points)} {}

  tetrahedra(Range0 &&tetras, const Range1 &points)
      : _tetras(std::move(tetras)), _points{points} {}

  auto tetras() const -> const Range0 & { return _tetras; }

  auto tetras() -> Range0 & { return _tetras; }

  auto points() const -> const Range1 & { return _points; }

  auto points() -> Range1 & { return _points; }

  auto begin() const -> const_iterator {
    return core::make_tetrahedron_range_iter(_tetras.begin(),
                                             unwrapped(_points));
  }

  auto begin() -> iterator {
    return core::make_tetrahedron_range_iter(_tetras.begin(),
                                             unwrapped(_points));
  }

  auto end() const -> const_iterator {
    return core::make_tetrahedron_range_iter(_tetras.end(), unwrapped(_points));
  }

  auto end() -> iterator {
    return core::make_tetrahedron_range_iter(_tetras.end(), unwrapped(_points));
  }

  auto size() const -> size_type { return _tetras.size(); }

  auto empty() const -> bool { return _tetras.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(begin() + size() - 1); }

  auto back() -> reference { return *(begin() + size() - 1); }

  auto operator[](size_type i) const -> const_reference {
    return *(begin() + i);
  }

  auto operator[](size_type i) -> reference { return *(begin() + i); }

private:
  Range0 _tetras;
  Range1 _points;
};

template <typename Range0, typename Range1>
auto make_tetrahedra(Range0 &&_tetras, Range1 &&_points) {
  return tetrahedra<std::decay_t<Range0>, std::decay_t<Range1>>{
      static_cast<Range0 &&>(_tetras), static_cast<Range1 &&>(_points)};
}

template <typename Range0, typename Range1>
auto has_tetras_policy(const tetrahedra<Range0, Range1> *) -> std::true_type;
auto has_tetras_policy(const void *) -> std::false_type;
} // namespace tf::core
