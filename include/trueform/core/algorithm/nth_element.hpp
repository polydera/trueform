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
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

namespace tf::core {

inline constexpr std::ptrdiff_t select_small_range = 24;
inline constexpr std::ptrdiff_t select_ninther_range = 128;
inline constexpr std::size_t select_shift_budget = 8;
inline constexpr int select_block = 64;

template <typename Compare> struct is_stock_order : std::false_type {};
template <typename T>
struct is_stock_order<std::less<T>> : std::true_type {};
template <typename T>
struct is_stock_order<std::greater<T>> : std::true_type {};

template <typename Iterator, typename Compare>
inline constexpr bool prefers_branchless_partition =
    is_stock_order<std::decay_t<Compare>>::value &&
    std::is_arithmetic_v<typename std::iterator_traits<Iterator>::value_type>;

template <typename Iterator, typename Compare>
auto order_pair(Iterator a, Iterator b, Compare &compare) -> void {
  if (compare(*b, *a))
    std::iter_swap(a, b);
}

template <typename Iterator, typename Compare>
auto order_triple(Iterator a, Iterator b, Iterator c, Compare &compare)
    -> void {
  core::order_pair(a, b, compare);
  core::order_pair(b, c, compare);
  core::order_pair(a, b, compare);
}

// With `Guarded` false the element before `first` stops every shift.
template <bool Guarded, typename Iterator, typename Compare>
auto insertion_sort_range(Iterator first, Iterator last, Compare &compare)
    -> void {
  using value_t = typename std::iterator_traits<Iterator>::value_type;
  if (first == last)
    return;
  for (Iterator cur = first + 1; cur != last; ++cur) {
    if (!compare(*cur, *(cur - 1)))
      continue;
    value_t held = std::move(*cur);
    Iterator hole = cur;
    do {
      *hole = std::move(*(hole - 1));
      --hole;
    } while ((!Guarded || hole != first) && compare(held, *(hole - 1)));
    *hole = std::move(held);
  }
}

// Sorts when the range is nearly sorted already; false once the shifts it
// performed exceed the budget.
template <typename Iterator, typename Compare>
auto sort_if_nearly_sorted(Iterator first, Iterator last, Compare &compare)
    -> bool {
  using value_t = typename std::iterator_traits<Iterator>::value_type;
  if (first == last)
    return true;
  std::size_t shifted = 0;
  for (Iterator cur = first + 1; cur != last; ++cur) {
    if (!compare(*cur, *(cur - 1)))
      continue;
    value_t held = std::move(*cur);
    Iterator hole = cur;
    do {
      *hole = std::move(*(hole - 1));
      --hole;
    } while (hole != first && compare(held, *(hole - 1)));
    *hole = std::move(held);
    shifted += std::size_t(cur - hole);
    if (shifted > select_shift_budget)
      return false;
  }
  return true;
}

// Moves the pivot to `*first`: the median of three, or of three medians of
// three on a large range.
template <typename Iterator, typename Compare>
auto place_pivot_first(Iterator first, Iterator last, Compare &compare)
    -> void {
  const auto half = (last - first) / 2;
  if (last - first > select_ninther_range) {
    core::order_triple(first, first + half, last - 1, compare);
    core::order_triple(first + 1, first + (half - 1), last - 2, compare);
    core::order_triple(first + 2, first + (half + 1), last - 3, compare);
    core::order_triple(first + (half - 1), first + half, first + (half + 1),
                       compare);
    std::iter_swap(first, first + half);
  } else {
    core::order_triple(first + half, first, last - 1, compare);
  }
}

// Marks the positions in `[first, first + count)` that are not below the
// pivot, as offsets from `first`.
template <typename Iterator, typename T, typename Compare>
auto mark_left_misplaced(Iterator first, int count, const T &pivot,
                         Compare &compare, unsigned char *offsets) -> int {
  int marked = 0;
  for (int i = 0; i < count; ++i) {
    offsets[marked] = static_cast<unsigned char>(i);
    marked += !compare(first[i], pivot);
  }
  return marked;
}

// Marks the positions in `[last - count, last)` that are below the pivot, as
// distances from `last`.
template <typename Iterator, typename T, typename Compare>
auto mark_right_misplaced(Iterator last, int count, const T &pivot,
                          Compare &compare, unsigned char *offsets) -> int {
  int marked = 0;
  for (int i = 0; i < count; ++i) {
    offsets[marked] = static_cast<unsigned char>(i + 1);
    marked += compare(*(last - (i + 1)), pivot);
  }
  return marked;
}

// Pairs the marked positions across the two ends. Equal mark counts exchange
// pair by pair; otherwise the pairs are rotated through one temporary.
template <typename Iterator>
auto exchange_marked(Iterator first, Iterator last, const unsigned char *left,
                     const unsigned char *right, int count, bool pairwise)
    -> void {
  using value_t = typename std::iterator_traits<Iterator>::value_type;
  if (count <= 0)
    return;
  if (pairwise) {
    for (int i = 0; i < count; ++i)
      std::iter_swap(first + left[i], last - right[i]);
    return;
  }
  Iterator l = first + left[0];
  Iterator r = last - right[0];
  value_t carried = std::move(*l);
  *l = std::move(*r);
  for (int i = 1; i < count; ++i) {
    l = first + left[i];
    *r = std::move(*l);
    r = last - right[i];
    *l = std::move(*r);
  }
  *r = std::move(carried);
}

// Elements equal to the pivot `*first` go right. Returns the pivot's final
// position and whether nothing had to move.
template <bool Branchless, typename Iterator, typename Compare>
auto partition_equals_right(Iterator first, Iterator last, Compare &compare)
    -> std::pair<Iterator, bool> {
  using value_t = typename std::iterator_traits<Iterator>::value_type;
  value_t pivot = std::move(*first);
  Iterator lo = first;
  Iterator hi = last;

  while (compare(*++lo, pivot))
    ;
  if (lo - 1 == first) {
    while (lo < hi && !compare(*--hi, pivot))
      ;
  } else {
    while (!compare(*--hi, pivot))
      ;
  }
  const bool untouched = lo >= hi;

  if constexpr (Branchless) {
    if (!untouched) {
      std::iter_swap(lo, hi);
      ++lo;
    }
    alignas(64) unsigned char left[select_block];
    alignas(64) unsigned char right[select_block];
    int n_left = 0, n_right = 0, at_left = 0, at_right = 0;

    while (hi - lo > 2 * select_block) {
      if (n_left == 0) {
        at_left = 0;
        n_left = core::mark_left_misplaced(lo, select_block, pivot, compare,
                                           left);
      }
      if (n_right == 0) {
        at_right = 0;
        n_right = core::mark_right_misplaced(hi, select_block, pivot, compare,
                                             right);
      }
      const int n = std::min(n_left, n_right);
      core::exchange_marked(lo, hi, left + at_left, right + at_right, n,
                            n_left == n_right);
      n_left -= n;
      n_right -= n;
      at_left += n;
      at_right += n;
      if (n_left == 0)
        lo += select_block;
      if (n_right == 0)
        hi -= select_block;
    }

    const int unknown =
        int(hi - lo) - ((n_left || n_right) ? select_block : 0);
    int left_span = unknown / 2;
    int right_span = unknown - left_span;
    if (n_right) {
      left_span = unknown;
      right_span = select_block;
    } else if (n_left) {
      left_span = select_block;
      right_span = unknown;
    }
    if (unknown && !n_left) {
      at_left = 0;
      n_left = core::mark_left_misplaced(lo, left_span, pivot, compare, left);
    }
    if (unknown && !n_right) {
      at_right = 0;
      n_right =
          core::mark_right_misplaced(hi, right_span, pivot, compare, right);
    }
    const int n = std::min(n_left, n_right);
    core::exchange_marked(lo, hi, left + at_left, right + at_right, n,
                          n_left == n_right);
    n_left -= n;
    n_right -= n;
    at_left += n;
    at_right += n;
    if (n_left == 0)
      lo += left_span;
    if (n_right == 0)
      hi -= right_span;

    if (n_left) {
      while (n_left--)
        std::iter_swap(lo + left[at_left + n_left], --hi);
      lo = hi;
    }
    if (n_right) {
      while (n_right--) {
        std::iter_swap(hi - right[at_right + n_right], lo);
        ++lo;
      }
      hi = lo;
    }
  } else {
    while (lo < hi) {
      std::iter_swap(lo, hi);
      while (compare(*++lo, pivot))
        ;
      while (!compare(*--hi, pivot))
        ;
    }
  }

  Iterator seat = lo - 1;
  *first = std::move(*seat);
  *seat = std::move(pivot);
  return {seat, untouched};
}

// Elements equal to the pivot `*first` go left. Returns the pivot's final
// position.
template <typename Iterator, typename Compare>
auto partition_equals_left(Iterator first, Iterator last, Compare &compare)
    -> Iterator {
  using value_t = typename std::iterator_traits<Iterator>::value_type;
  value_t pivot = std::move(*first);
  Iterator lo = first;
  Iterator hi = last;

  while (compare(pivot, *--hi))
    ;
  if (hi + 1 == last) {
    while (lo < hi && !compare(pivot, *++lo))
      ;
  } else {
    while (!compare(pivot, *++lo))
      ;
  }
  while (lo < hi) {
    std::iter_swap(lo, hi);
    while (compare(pivot, *--hi))
      ;
    while (!compare(pivot, *++lo))
      ;
  }

  *first = std::move(*hi);
  *hi = std::move(pivot);
  return hi;
}

// Swaps a few elements near both ends of each side of a lopsided partition,
// so the pattern that produced it does not repeat.
template <typename Iterator>
auto scatter_lopsided(Iterator first, Iterator last, Iterator seat) -> void {
  const auto n_left = seat - first;
  const auto n_right = last - (seat + 1);
  if (n_left >= select_small_range) {
    const auto q = n_left / 4;
    std::iter_swap(first, first + q);
    std::iter_swap(seat - 1, seat - q);
    if (n_left > select_ninther_range) {
      std::iter_swap(first + 1, first + (q + 1));
      std::iter_swap(first + 2, first + (q + 2));
      std::iter_swap(seat - 2, seat - (q + 1));
      std::iter_swap(seat - 3, seat - (q + 2));
    }
  }
  if (n_right >= select_small_range) {
    const auto q = n_right / 4;
    std::iter_swap(seat + 1, seat + (1 + q));
    std::iter_swap(last - 1, last - q);
    if (n_right > select_ninther_range) {
      std::iter_swap(seat + 2, seat + (2 + q));
      std::iter_swap(seat + 3, seat + (3 + q));
      std::iter_swap(last - 2, last - (1 + q));
      std::iter_swap(last - 3, last - (2 + q));
    }
  }
}

// Bounds the worst case at O(n log k): a max-heap of the k + 1 smallest seen.
template <typename Iterator, typename Compare>
auto heap_select(Iterator first, Iterator nth, Iterator last, Compare &compare)
    -> void {
  Iterator heap_end = nth + 1;
  std::make_heap(first, heap_end, compare);
  for (Iterator it = heap_end; it != last; ++it) {
    if (!compare(*it, *first))
      continue;
    std::pop_heap(first, heap_end, compare);
    std::iter_swap(nth, it);
    std::push_heap(first, heap_end, compare);
  }
  std::iter_swap(first, nth);
}

template <bool Branchless, typename Iterator, typename Compare>
auto pdqselect(Iterator first, Iterator nth, Iterator last, Compare compare)
    -> void {
  if (nth == last)
    return;
  int bad_partitions = 0;
  for (auto n = last - first; n > 1; n >>= 1)
    ++bad_partitions;
  bool leftmost = true;

  while (true) {
    const auto size = last - first;
    if (size < select_small_range) {
      if (leftmost)
        core::insertion_sort_range<true>(first, last, compare);
      else
        core::insertion_sort_range<false>(first, last, compare);
      return;
    }

    core::place_pivot_first(first, last, compare);

    if (!leftmost && !compare(*(first - 1), *first)) {
      first = core::partition_equals_left(first, last, compare) + 1;
      continue;
    }

    auto [seat, untouched] =
        core::partition_equals_right<Branchless>(first, last, compare);
    const auto n_left = seat - first;
    const auto n_right = last - (seat + 1);

    if (n_left < size / 8 || n_right < size / 8) {
      if (--bad_partitions == 0) {
        core::heap_select(first, nth, last, compare);
        return;
      }
      core::scatter_lopsided(first, last, seat);
    } else if (untouched &&
               core::sort_if_nearly_sorted(first, seat, compare) &&
               core::sort_if_nearly_sorted(seat + 1, last, compare)) {
      return;
    }

    if (seat < nth) {
      first = seat + 1;
      leftmost = false;
    } else {
      last = seat;
    }
  }
}

} // namespace tf::core

namespace tf {

/// @ingroup core_algorithms
/// @brief Place the element that belongs at `nth` in sorted order exactly,
/// with every element before it not greater and every element after it not
/// less, the two sides unordered — the `std::nth_element` contract. The
/// partition is branchless under `std::less` or `std::greater` on arithmetic
/// keys and branchy otherwise.
template <typename Iterator, typename Compare>
auto nth_element(Iterator first, Iterator nth, Iterator last, Compare compare)
    -> void {
  core::pdqselect<core::prefers_branchless_partition<Iterator, Compare>>(
      std::move(first), std::move(nth), std::move(last), std::move(compare));
}

/// @ingroup core_algorithms
/// @brief @ref tf::nth_element under `std::less`.
/// @overload
template <typename Iterator>
auto nth_element(Iterator first, Iterator nth, Iterator last) -> void {
  tf::nth_element(
      std::move(first), std::move(nth), std::move(last),
      std::less<typename std::iterator_traits<Iterator>::value_type>());
}

} // namespace tf
