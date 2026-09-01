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
#include "../checked.hpp"
#include "../grain.hpp"
#include "tbb/parallel_for.h"

namespace tf {
/// @ingroup core_algorithms
/// @brief Applies a function to each element of a range in parallel.
///
/// Uses TBB to split the range and apply the function concurrently.
///
/// @tparam Range A range type supporting `begin()` and `end()`.
/// @tparam Func A callable taking a reference to an element of the range.
/// @param r The range to iterate over.
/// @param f The function to apply to each element.
///
/// @note Parallel equivalent of:
/// @code
/// for (auto &elem : r) f(elem);
/// @endcode
///
/// @see @ref parallel_for
template <typename Range, typename Func>
auto parallel_for_each(Range &&r, Func &&f) -> void {
  auto first = r.begin();
  auto last = r.end();
  using Iterator = decltype(first);
  tbb::parallel_for(
      tbb::blocked_range<Iterator>(first, last),
      [f = static_cast<Func &&>(f)](const tbb::blocked_range<Iterator> &range) {
        for (Iterator it = range.begin(); it != range.end(); ++it) {
          if constexpr (std::is_integral<Iterator>::value)
            f(it);
          else
            f(*it);
        }
      });
}

/// @ingroup core_algorithms
/// @brief Applies a function to each element in parallel with a minimum
/// chunk size.
///
/// Like @ref parallel_for_each, but the range is never split below
/// `grain.value` elements per task, so each task keeps a run of
/// consecutive elements (bounded scheduling overhead, cache locality
/// across neighboring elements).
///
/// @tparam Range A range type supporting `begin()` and `end()`.
/// @tparam Func A callable taking a reference to an element of the range.
/// @param r The range to iterate over.
/// @param f The function to apply to each element.
/// @param grain Minimum number of consecutive elements per task.
///
/// @see @ref parallel_for_each
template <typename Range, typename Func>
auto parallel_for_each(Range &&r, Func &&f, tf::grain_t grain) -> void {
  auto first = r.begin();
  auto last = r.end();
  using Iterator = decltype(first);
  tbb::parallel_for(
      tbb::blocked_range<Iterator>(first, last, grain.value),
      [f = static_cast<Func &&>(f)](const tbb::blocked_range<Iterator> &range) {
        for (Iterator it = range.begin(); it != range.end(); ++it) {
          if constexpr (std::is_integral<Iterator>::value)
            f(it);
          else
            f(*it);
        }
      });
}

/// @ingroup core_algorithms
/// @brief Applies a function to each element with per-task state.
///
/// Each parallel task receives a copy of `state`, enabling reusable block-local
/// scratch or accumulation.
///
/// @tparam Range A range type supporting `begin()` and `end()`.
/// @tparam Func A callable taking an element reference and state.
/// @tparam State The per-task state type (copied per task).
/// @param r The range to iterate over.
/// @param f The function to apply, signature `f(element, state)`.
/// @param state The initial state copied to each task.
///
/// @see @ref parallel_for
template <typename Range, typename Func, typename State>
auto parallel_for_each(Range &&r, Func &&f, State state) -> void {
  auto first = r.begin();
  auto last = r.end();
  using Iterator = decltype(first);
  tbb::parallel_for(tbb::blocked_range<Iterator>(first, last),
                    [&state, f = static_cast<Func &&>(f)](
                        const tbb::blocked_range<Iterator> &range) {
                      auto l_state = state;
                      for (Iterator it = range.begin(); it != range.end();
                           ++it) {
                        if constexpr (std::is_integral<Iterator>::value)
                          f(it, l_state);
                        else
                          f(*it, l_state);
                      }
                    });
}

/// @ingroup core_algorithms
/// @brief Applies a function to each element with checked execution.
///
/// Falls back to sequential execution for ranges smaller than 1000 elements
/// to avoid parallel overhead on small workloads.
///
/// @tparam Range A range type supporting `begin()`, `end()`, and `size()`.
/// @tparam Func A callable taking a reference to an element of the range.
/// @param r The range to iterate over.
/// @param f The function to apply to each element.
///
/// @see @ref parallel_for_each
template <typename Range, typename Func>
auto parallel_for_each(Range &&r, Func &&f, tf::checked_t c) -> void {
  if (std::size_t(r.size()) < c.serial_below) {
    for (auto &&e : r)
      f(e);
  } else
    return parallel_for_each(r, static_cast<Func &&>(f));
}

/// @ingroup core_algorithms
/// @brief Applies a function with per-task state and checked execution.
///
/// Combines per-task state with checked execution, falling back to
/// sequential for ranges smaller than 1000 elements.
///
/// @tparam Range A range type supporting `begin()`, `end()`, and `size()`.
/// @tparam Func A callable taking an element reference and state.
/// @tparam State The per-task state type (copied per task).
/// @param r The range to iterate over.
/// @param f The function to apply, signature `f(element, state)`.
/// @param state The initial state copied to each task.
///
/// @see @ref parallel_for_each
template <typename Range, typename Func, typename State>
auto parallel_for_each(Range &&r, Func &&f, State state, tf::checked_t c)
    -> void {
  if (std::size_t(r.size()) < c.serial_below) {
    for (auto &&e : r)
      f(e, state);
  } else
    return parallel_for_each(r, static_cast<Func &&>(f), std::move(state));
}
} // namespace tf
