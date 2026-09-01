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
#include "../../core/buffer.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt {

template <typename Index> struct retained_delaunay_half_edge {
  Index vertex;
  Index prev;
  Index next;
  bool boundary;
  bool delaunay;
  std::uint8_t constrained;
};

/// Hot origin-ring fields and cold constraint state live in separate flat
/// carriers. Initial Delaunay construction touches only the three Index lanes;
/// retained state is initialized once, linearly, after topology construction.
template <typename Index> class retained_half_edges {
public:
  using value_type = retained_delaunay_half_edge<Index>;

  struct reference {
    Index &vertex;
    Index &prev;
    Index &next;
    std::uint8_t &boundary;
    std::uint8_t &delaunay;
    std::uint8_t &constrained;

    auto operator=(const value_type &edge) -> reference & {
      vertex = edge.vertex;
      prev = edge.prev;
      next = edge.next;
      boundary = static_cast<std::uint8_t>(edge.boundary);
      delaunay = static_cast<std::uint8_t>(edge.delaunay);
      constrained = edge.constrained;
      return *this;
    }
  };

  struct const_reference {
    const Index &vertex;
    const Index &prev;
    const Index &next;
    const std::uint8_t &boundary;
    const std::uint8_t &delaunay;
    const std::uint8_t &constrained;
  };

  retained_half_edges() = default;
  retained_half_edges(const retained_half_edges &) = default;
  auto operator=(const retained_half_edges &)
      -> retained_half_edges & = default;

  retained_half_edges(retained_half_edges &&other) noexcept
      : _vertices(std::move(other._vertices)),
        _previous(std::move(other._previous)), _next(std::move(other._next)),
        _boundary(std::move(other._boundary)),
        _delaunay(std::move(other._delaunay)),
        _constrained(std::move(other._constrained)), _size(other._size),
        _capacity(other._capacity) {
    other._size = 0;
    other._capacity = 0;
  }

  auto operator=(retained_half_edges &&other) noexcept
      -> retained_half_edges & {
    if (this == &other)
      return *this;
    _vertices = std::move(other._vertices);
    _previous = std::move(other._previous);
    _next = std::move(other._next);
    _boundary = std::move(other._boundary);
    _delaunay = std::move(other._delaunay);
    _constrained = std::move(other._constrained);
    _size = other._size;
    _capacity = other._capacity;
    other._size = 0;
    other._capacity = 0;
    return *this;
  }

  auto clear() -> void { _size = 0; }

  auto reserve(std::size_t size) -> void {
    if (size <= _capacity)
      return;
    _vertices.reallocate(size);
    _previous.reallocate(size);
    _next.reallocate(size);
    _boundary.reallocate(size);
    _delaunay.reallocate(size);
    _constrained.reallocate(size);
    _capacity = size;
  }

  auto reallocate(std::size_t size) -> void {
    if (size > _capacity)
      reserve(std::max(size, _capacity * 2));
    _size = size;
  }

  auto append_pair() -> std::size_t {
    const std::size_t first = _size;
    reallocate(_size + 2);
    return first;
  }

  auto push_back(const value_type &edge) -> void {
    const std::size_t index = _size;
    reallocate(_size + 1);
    (*this)[index] = edge;
  }

  auto size() const -> std::size_t { return _size; }

  auto operator[](std::size_t index) -> reference {
    return {_vertices[index], _previous[index], _next[index],
            _boundary[index], _delaunay[index], _constrained[index]};
  }

  auto operator[](std::size_t index) const -> const_reference {
    return {_vertices[index], _previous[index], _next[index],
            _boundary[index], _delaunay[index], _constrained[index]};
  }

  auto initialize_retained_state(Index none) -> void {
    for (std::size_t i = 0; i < _size; ++i)
      _boundary[i] = static_cast<std::uint8_t>(_vertices[i] == none);
    std::fill(_delaunay.begin(), _delaunay.begin() + std::ptrdiff_t(_size),
              std::uint8_t(1));
    std::fill(_constrained.begin(),
              _constrained.begin() + std::ptrdiff_t(_size), std::uint8_t(0));
  }

private:
  tf::buffer<Index> _vertices;
  tf::buffer<Index> _previous;
  tf::buffer<Index> _next;
  tf::buffer<std::uint8_t> _boundary;
  tf::buffer<std::uint8_t> _delaunay;
  tf::buffer<std::uint8_t> _constrained;
  std::size_t _size = 0;
  std::size_t _capacity = 0;
};

} // namespace tf::topology::cdt
