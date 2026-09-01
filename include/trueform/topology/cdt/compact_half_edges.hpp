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
#include <utility>

namespace tf::topology::cdt {

/// Three-array dart store for topology kernels that only retain origin rings.
/// Its proxy has the same vertex/prev/next field surface as a twelve-byte AoS
/// half-edge, letting the generic divide-and-conquer builder stay storage-free.
template <typename Index> class compact_half_edges {
public:
  struct reference {
    Index &vertex;
    Index &prev;
    Index &next;
  };

  struct const_reference {
    const Index &vertex;
    const Index &prev;
    const Index &next;
  };

  compact_half_edges() = default;
  compact_half_edges(const compact_half_edges &) = default;
  auto operator=(const compact_half_edges &) -> compact_half_edges & = default;

  compact_half_edges(compact_half_edges &&other) noexcept
      : _vertices(std::move(other._vertices)),
        _previous(std::move(other._previous)), _next(std::move(other._next)),
        _size(other._size), _capacity(other._capacity) {
    other._size = 0;
    other._capacity = 0;
  }

  auto operator=(compact_half_edges &&other) noexcept -> compact_half_edges & {
    if (this == &other)
      return *this;
    _vertices = std::move(other._vertices);
    _previous = std::move(other._previous);
    _next = std::move(other._next);
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
    // The backing buffers represent arena capacity, not logical dart count.
    // Keeping the logical cursor separately makes the append-only hot path a
    // single capacity branch instead of three buffer growth checks.
    _vertices.reallocate(size);
    _previous.reallocate(size);
    _next.reallocate(size);
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

  auto size() const -> std::size_t { return _size; }

  auto operator[](std::size_t index) -> reference {
    return {_vertices[index], _previous[index], _next[index]};
  }

  auto operator[](std::size_t index) const -> const_reference {
    return {_vertices[index], _previous[index], _next[index]};
  }

private:
  tf::buffer<Index> _vertices;
  tf::buffer<Index> _previous;
  tf::buffer<Index> _next;
  std::size_t _size = 0;
  std::size_t _capacity = 0;
};

} // namespace tf::topology::cdt
