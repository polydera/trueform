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

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include <type_traits>

namespace tf {

/// @ingroup core_algorithms
/// @brief Tag type for cancellation-isolated parallel execution.
struct isolated_t {};

/// @ingroup core_algorithms
/// @brief Selects cancellation-isolated parallel execution.
inline constexpr isolated_t isolated{};

/// @ingroup core_algorithms
/// @brief Applies a function to each element without inheriting cancellation.
///
/// Use this operation when every element must be visited even if the caller's
/// task group has already been cancelled.
template <typename Range, typename Func>
auto parallel_for_each(Range &&r, Func &&f, tf::isolated_t) -> void {
  auto first = r.begin();
  auto last = r.end();
  using Iterator = decltype(first);
  tbb::task_group_context context(tbb::task_group_context::isolated);
  tbb::parallel_for(
      tbb::blocked_range<Iterator>(first, last),
      [f = static_cast<Func &&>(f)](const tbb::blocked_range<Iterator> &range) {
        for (Iterator it = range.begin(); it != range.end(); ++it) {
          if constexpr (std::is_integral<Iterator>::value)
            f(it);
          else
            f(*it);
        }
      },
      context);
}

} // namespace tf
