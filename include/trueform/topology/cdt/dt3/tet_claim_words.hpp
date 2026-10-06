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
#include "../../../core/range.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_claim_word.hpp"
#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// One claim word per cell row. `tf::allocate` returns raw storage, not live
/// atomics, so every word is placement-constructed when the storage is
/// sized, and a copy constructs its words from the source's values.
class tet_claim_words {
public:
  tet_claim_words() = default;

  tet_claim_words(tet_claim_words &&other) noexcept
      : _words(std::move(other._words)),
        _size(std::exchange(other._size, 0)),
        _capacity(std::exchange(other._capacity, 0)) {}

  auto operator=(tet_claim_words &&other) noexcept -> tet_claim_words & {
    _words = std::move(other._words);
    _size = std::exchange(other._size, 0);
    _capacity = std::exchange(other._capacity, 0);
    return *this;
  }

  tet_claim_words(const tet_claim_words &other) { copy_from(other); }

  auto operator=(const tet_claim_words &other) -> tet_claim_words & {
    if (this != &other)
      copy_from(other);
    return *this;
  }

  auto allocate_and_initialize(std::size_t n, tet_claim_word value) -> void {
    reserve(n);
    _size = n;
    tf::parallel_for_each(
        words(),
        [value](std::atomic<tet_claim_word> &word) {
          ::new (static_cast<void *>(&word)) std::atomic<tet_claim_word>(value);
        },
        tf::checked);
  }

  auto operator[](std::size_t i) -> std::atomic<tet_claim_word> & {
    return _words[i];
  }

  auto operator[](std::size_t i) const -> const std::atomic<tet_claim_word> & {
    return _words[i];
  }

  auto size() const -> std::size_t { return _size; }

  auto clear() -> void { _size = 0; }

private:
  auto words() -> tf::range<std::atomic<tet_claim_word> *, tf::dynamic_size> {
    return tf::make_range(_words.get(), _size);
  }

  auto reserve(std::size_t n) -> void {
    if (n <= _capacity)
      return;
    _words.reset(tf::allocate<std::atomic<tet_claim_word>>(n));
    _capacity = n;
  }

  auto copy_from(const tet_claim_words &other) -> void {
    reserve(other._size);
    _size = other._size;
    tf::parallel_for_each(
        tf::make_sequence_range(_size),
        [this, &other](std::size_t i) {
          ::new (static_cast<void *>(&_words[i])) std::atomic<tet_claim_word>(
              other._words[i].load(std::memory_order_relaxed));
        },
        tf::checked);
  }

  struct words_deleter {
    auto operator()(void *p) const noexcept -> void {
      tf::deallocate<std::atomic<tet_claim_word>>(p);
    }
  };

  std::unique_ptr<std::atomic<tet_claim_word>[], words_deleter> _words;
  std::size_t _size = 0;
  std::size_t _capacity = 0;
};

} // namespace tf::topology::cdt::dt3
