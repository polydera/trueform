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
#include "../cache_aligned_slot.hpp"
#include "../checked.hpp"
#include "../memory.hpp"
#include "../range.hpp"
#include "./block_reduce_graph.hpp"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
namespace tf {

/// @ingroup core_algorithms
/// @brief Parallel blocked reduction with custom local and global aggregation.
///
/// Divides the input range into blocks, processes each block in parallel
/// using the `task` function to accumulate into a block-local result, then
/// aggregates all local results into the global result using `aggregate`.
///
/// The aggregation happens sequentially to ensure thread-safe access
/// to the global result.
///
/// @tparam Range The input range type.
/// @tparam Result The global result type.
/// @tparam LocalResult The block-local result type, copied once per work block.
/// @tparam F0 Block processing function: `void(block_range, LocalResult&)`.
/// @tparam F1 Aggregation function: `void(const LocalResult&, Result&)`.
/// @param data The input range to reduce.
/// @param result The global result (accumulated into).
/// @param local_result Template copied to initialize each block-local result.
/// @param task Function to process each block.
/// @param aggregate Function to merge local into global result.
/// @param n_blocks Number of blocks (default: 5x hardware concurrency).
template <typename Range, typename Result, typename LocalResult, typename F0,
          typename F1>
auto blocked_reduce(const Range &data, Result &&result,
                    LocalResult local_result, F0 task, F1 aggregate,
                    std::size_t n_blocks = std::thread::hardware_concurrency() *
                                           5) {
  if (data.size() == 0)
    return;
  if (std::thread::hardware_concurrency() == 1)
    n_blocks = 1;
  if (n_blocks == 1) {
    task(tf::make_range(data), local_result);
    aggregate(local_result, result);
    return;
  }
  auto step = data.size() / n_blocks;
  step = std::max(decltype(step)(1), step);
  auto n_tasks = (data.size() + step - 1) / step;

  core::std_vector<core::cache_aligned_slot<std::optional<LocalResult>>> locals(
      static_cast<std::size_t>(n_tasks));
  core::reduce_call_site<Range, std::remove_reference_t<Result>, LocalResult,
                         F0, F1>
      call_site{&data, std::size_t(step), &local_result, locals.data(),
                &task, &aggregate,        &result};
  core::run_reduce_graph(&call_site, call_site.bodies(),
                         static_cast<std::size_t>(n_tasks));
}

/// @ingroup core_algorithms
/// @brief Parallel blocked reduction using the result as the local template.
///
/// Convenience overload that uses a copy of the result as the local
/// accumulator template.
template <typename Range, typename Result, typename F0, typename F1>
auto blocked_reduce(const Range &data, Result &&result, F0 task, F1 aggregate,
                    std::size_t n_blocks = std::thread::hardware_concurrency() *
                                           5) {
  auto local_result = result;
  return blocked_reduce(data, static_cast<Result &&>(result), local_result,
                        std::move(task), std::move(aggregate), n_blocks);
}

/// @ingroup core_algorithms
/// @brief Checked blocked reduction: one block below the parallel entry cost.
///
/// The serial path is the single-block path — the task sees the whole range
/// and the aggregate runs once — so no dependency graph is built and nothing
/// a caller may rely on changes.
template <typename Range, typename Result, typename LocalResult, typename F0,
          typename F1>
auto blocked_reduce(const Range &data, Result &&result,
                    LocalResult local_result, F0 task, F1 aggregate,
                    tf::checked_t c) {
  if (std::size_t(data.size()) < c.serial_below)
    return blocked_reduce(data, static_cast<Result &&>(result),
                          std::move(local_result), std::move(task),
                          std::move(aggregate), std::size_t(1));
  return blocked_reduce(data, static_cast<Result &&>(result),
                        std::move(local_result), std::move(task),
                        std::move(aggregate));
}

/// @ingroup core_algorithms
/// @brief Checked blocked reduction with the result as the local template.
/// @overload
template <typename Range, typename Result, typename F0, typename F1>
auto blocked_reduce(const Range &data, Result &&result, F0 task, F1 aggregate,
                    tf::checked_t c) {
  auto local_result = result;
  return blocked_reduce(data, static_cast<Result &&>(result), local_result,
                        std::move(task), std::move(aggregate), c);
}
} // namespace tf
