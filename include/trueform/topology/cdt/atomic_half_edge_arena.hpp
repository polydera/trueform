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
#include "./delaunay_execution_tuning.hpp"
#include <atomic>
#include <cassert>
#include <cstddef>

namespace tf::topology::cdt {

/// Shared block allocation authority for the parallel half-edge build. Darts
/// need stable global identities while a subtree mutates its topology, but its
/// exact output count is only known afterward. A local cursor therefore
/// amortizes each relaxed atomic claim over a large contiguous block.
template <typename Edges, std::size_t BlockSize =
                              delaunay_execution_tuning::edge_arena_block_darts>
class atomic_half_edge_arena {
  static_assert(BlockSize >= 2 && (BlockSize & 1) == 0);

public:
  struct capacity_exceeded {};

  class cursor {
  public:
    cursor(const cursor &) = delete;
    cursor(cursor &&) = delete;
    auto operator=(const cursor &) -> cursor & = delete;
    auto operator=(cursor &&) -> cursor & = delete;

    auto append_pair() -> std::size_t {
      if (_next + 2 > _end)
        acquire_block();
      const std::size_t first = _next;
      _next += 2;
      return first;
    }

    decltype(auto) operator[](std::size_t index) {
      assert(index < _arena->_capacity);
      return (*_arena->_edges)[index];
    }

    decltype(auto) operator[](std::size_t index) const {
      assert(index < _arena->_capacity);
      return static_cast<const Edges &>(*_arena->_edges)[index];
    }

    template <typename Index> auto finish(Index none) -> void {
      // Unused block tails remain inside the committed high-water range.
      for (std::size_t index = _next; index < _end; ++index)
        (*_arena->_edges)[index].vertex = none;
    }

  private:
    friend class atomic_half_edge_arena;

    explicit cursor(atomic_half_edge_arena &arena) : _arena(&arena) {}

    auto acquire_block() -> void {
      const auto block = _arena->acquire_block();
      _next = block.first;
      _end = block.last;
    }

    atomic_half_edge_arena *_arena;
    std::size_t _next = 0;
    std::size_t _end = 0;
  };

  atomic_half_edge_arena(Edges &edges, std::size_t capacity)
      : _edges(&edges), _capacity(capacity) {
    _edges->clear();
    _edges->reserve(capacity);
  }

  auto make_cursor() -> cursor { return cursor(*this); }

  /// Publish the claimed high-water range after every cursor has finished.
  auto commit() -> void {
    const std::size_t claimed = _next.load(std::memory_order_relaxed);
    const std::size_t high_water = claimed < _capacity ? claimed : _capacity;
    _edges->reallocate(high_water & ~std::size_t(1));
  }

private:
  struct block {
    std::size_t first;
    std::size_t last;
  };

  auto acquire_block() -> block {
    const std::size_t first =
        _next.fetch_add(BlockSize, std::memory_order_relaxed);
    if (first >= _capacity)
      throw capacity_exceeded{};
    const std::size_t available = _capacity - first;
    return {first, first + (available < BlockSize ? available : BlockSize)};
  }

  Edges *_edges;
  std::size_t _capacity;
  std::atomic<std::size_t> _next{0};
};

} // namespace tf::topology::cdt
