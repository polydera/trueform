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
#include "./parallel_for.hpp"
#include <atomic>

namespace tf {

/// @ingroup core_algorithms
/// @brief Checks if any element in a range satisfies a predicate, in parallel.
///
/// Uses parallel_for to split the range and check concurrently. Returns early
/// once a matching element is found.
///
/// @tparam Range A range type supporting `begin()` and `end()`.
/// @tparam Pred A callable returning bool for each element.
/// @param r The range to search.
/// @param pred The predicate to test each element.
/// @return `true` if any element satisfies the predicate.
template <typename Range, typename Pred>
auto parallel_contains(Range &&r, Pred &&pred) -> bool {
  std::atomic_bool found{false};

  tf::parallel_for(r, [&found, &pred](auto first, auto last) {
    for (auto it = first; it != last; ++it) {
      if (found.load(std::memory_order_relaxed))
        return;
      if (pred(*it)) {
        found.store(true, std::memory_order_relaxed);
        return;
      }
    }
  });

  return found.load();
}

/// @ingroup core_algorithms
/// @brief Checks if any element satisfies a predicate, with checked execution.
///
/// Falls back to sequential execution for ranges smaller than 1000 elements
/// to avoid parallel overhead on small workloads.
///
/// @tparam Range A range type supporting `begin()`, `end()`, and `size()`.
/// @tparam Pred A callable returning bool for each element.
/// @param r The range to search.
/// @param pred The predicate to test each element.
/// @return `true` if any element satisfies the predicate.
template <typename Range, typename Pred>
auto parallel_contains(Range &&r, Pred &&pred, tf::checked_t) -> bool {
  if (r.size() < 1000) {
    for (auto &&e : r)
      if (pred(e))
        return true;
    return false;
  }
  return parallel_contains(r, static_cast<Pred &&>(pred));
}

} // namespace tf
