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
#include "../../../core/views/drop.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_claim_word.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// One claim word per carrier: a cell row the build's streams claim, or a
/// site or segment the smallest facet naming it claims
/// (@ref make_tet_first_facet_patches). `tf::allocate` returns raw storage, not live
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
    reset(n);
    _size = n;
    tf::parallel_for_each(
        words(),
        [value](std::atomic<tet_claim_word> &word) {
          ::new (static_cast<void *>(&word)) std::atomic<tet_claim_word>(value);
        },
        tf::checked);
  }

  /// Size the lane to `n`, keeping the first words and constructing the
  /// appended ones as `value`; capacity grows geometrically, so a lane grown
  /// round by round moves each word a bounded number of times.
  auto reallocate_and_initialize(std::size_t n, tet_claim_word value) -> void {
    const std::size_t kept = std::min(_size, n);
    if (n > _capacity)
      reserve(std::max(n, 2 * _capacity));
    _size = n;
    tf::parallel_for_each(
        tf::drop(words(), kept),
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

  /// Grow the capacity to `n`, keeping the words.
  auto reserve(std::size_t n) -> void {
    if (n <= _capacity)
      return;
    std::unique_ptr<std::atomic<tet_claim_word>[], words_deleter> words(
        tf::allocate<std::atomic<tet_claim_word>>(n));
    auto *from = _words.get();
    auto *to = words.get();
    tf::parallel_for_each(
        tf::make_sequence_range(_size),
        [from, to](std::size_t i) {
          ::new (static_cast<void *>(&to[i])) std::atomic<tet_claim_word>(
              from[i].load(std::memory_order_relaxed));
        },
        tf::checked);
    _words = std::move(words);
    _capacity = n;
  }

  auto size() const -> std::size_t { return _size; }

  auto clear() -> void { _size = 0; }

private:
  auto words() -> tf::range<std::atomic<tet_claim_word> *, tf::dynamic_size> {
    return tf::make_range(_words.get(), _size);
  }

  auto reset(std::size_t n) -> void {
    if (n <= _capacity)
      return;
    _words.reset(tf::allocate<std::atomic<tet_claim_word>>(n));
    _capacity = n;
  }

  auto copy_from(const tet_claim_words &other) -> void {
    reset(other._size);
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
