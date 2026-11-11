/*
 * Copyright (c) 2026 XLAB
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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/memory.hpp"
#include "../../../core/views/sequence_range.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// One cell holding each site, written wherever a cell is written, so a
/// site's star is entered without a walk. Concurrent insertions sharing a
/// site write its slot together, any of their cells answering, so every
/// slot is a relaxed atomic; a bulk record lowers each slot to the lowest
/// cell holding the site (@ref lower), the same at any worker count, so a
/// recovery that reads its rings from the slot runs one stream.
///
/// THE SLOT BELONGS, FOR THE ROUND, TO THE REGION OF ITS OPENING CELL
/// (@ref run_tet_region_round). A writer stores into a slot only where the
/// cell the slot names carries the writer's own mark, so one region writes
/// one slot and no other overwrites it; a reader uses the answer only where
/// that cell is its own — then the value came from the opening or from its
/// own region — and a slot of another region stops the task, the same
/// verdict at any worker count because the slot's owner is fixed by the
/// opening. The barrier's lowering restates every written site's slot
/// afterwards, so the round's product is the same at any width.
///
/// `tf::allocate` returns raw storage, so slots are placement-constructed
/// as they are sized, and a copy constructs its slots from the source's
/// values.
template <typename Index> class tet_site_cells {
public:
  tet_site_cells() = default;

  tet_site_cells(tet_site_cells &&other) noexcept
      : _slots(std::move(other._slots)), _size(std::exchange(other._size, 0)),
        _capacity(std::exchange(other._capacity, 0)) {}

  auto operator=(tet_site_cells &&other) noexcept -> tet_site_cells & {
    _slots = std::move(other._slots);
    _size = std::exchange(other._size, 0);
    _capacity = std::exchange(other._capacity, 0);
    return *this;
  }

  tet_site_cells(const tet_site_cells &other) { copy_from(other); }

  auto operator=(const tet_site_cells &other) -> tet_site_cells & {
    if (this != &other)
      copy_from(other);
    return *this;
  }

  /// Grow or shrink to `n` sites, the kept slots keeping their cells and
  /// the new ones holding `none`.
  auto resize(std::size_t n, Index none) -> void {
    if (n > _capacity) {
      std::size_t capacity = std::max(n, 2 * _capacity);
      slots_ptr grown(tf::allocate<std::atomic<Index>>(capacity));
      for (std::size_t i = 0; i < _size; ++i)
        ::new (static_cast<void *>(&grown[i]))
            std::atomic<Index>(_slots[i].load(std::memory_order_relaxed));
      _slots = std::move(grown);
      _capacity = capacity;
    }
    for (std::size_t i = _size; i < n; ++i)
      ::new (static_cast<void *>(&_slots[i])) std::atomic<Index>(none);
    _size = n;
  }

  auto store(std::size_t site, Index cell) -> void {
    _slots[site].store(cell, std::memory_order_relaxed);
  }
  /// Lower the slot to `cell` where `cell` is below what it holds, `none`
  /// counting as above every cell.
  auto lower(std::size_t site, Index cell, Index none) -> void {
    Index held = _slots[site].load(std::memory_order_relaxed);
    while ((held == none || cell < held) &&
           !_slots[site].compare_exchange_weak(held, cell,
                                               std::memory_order_relaxed))
      ;
  }

  auto load(std::size_t site) const -> Index {
    return _slots[site].load(std::memory_order_relaxed);
  }

  /// Read every slot through `map`, old cell to new.
  template <typename Map> auto remap(const Map &map, Index none) -> void {
    tf::parallel_for_each(
        tf::make_sequence_range(_size),
        [this, &map, none](std::size_t i) {
          const Index cell = _slots[i].load(std::memory_order_relaxed);
          _slots[i].store(cell == none ? none : map[std::size_t(cell)],
                          std::memory_order_relaxed);
        },
        tf::checked);
  }

  auto size() const -> std::size_t { return _size; }

  auto clear() -> void { _size = 0; }

private:
  struct slots_deleter {
    auto operator()(void *p) const noexcept -> void {
      tf::deallocate<std::atomic<Index>>(p);
    }
  };
  using slots_ptr = std::unique_ptr<std::atomic<Index>[], slots_deleter>;

  auto copy_from(const tet_site_cells &other) -> void {
    if (other._size > _capacity) {
      _slots.reset(tf::allocate<std::atomic<Index>>(other._size));
      _capacity = other._size;
    }
    _size = other._size;
    for (std::size_t i = 0; i < _size; ++i)
      ::new (static_cast<void *>(&_slots[i]))
          std::atomic<Index>(other._slots[i].load(std::memory_order_relaxed));
  }

  slots_ptr _slots;
  std::size_t _size = 0;
  std::size_t _capacity = 0;
};

} // namespace tf::topology::cdt::dt3
