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
namespace core {

/// @brief The chunk stride `blocked_reduce_sequenced_aggregate` partitions by.
///
/// Its chunks are contiguous runs of this length, so two positions a stride
/// apart are two positions in different chunks. A caller that orders the work
/// it hands the primitive reads the partition from here.
///
/// @param n Number of items.
/// @param n_blocks Number of parallel blocks.
/// @return The length of one chunk.
inline auto blocked_sequenced_chunk_stride(
    std::size_t n,
    std::size_t n_blocks = std::thread::hardware_concurrency() * 5)
    -> std::size_t {
  if (std::thread::hardware_concurrency() == 1)
    n_blocks = 1;
  if (n_blocks == 1)
    return n;
  return std::max(std::size_t(1), n / n_blocks);
}

} // namespace core

/// @ingroup core_algorithms
/// @brief Parallel reduction with ordered aggregation.
///
/// Performs parallel work on blocks but aggregates results in input-block order.
/// Useful when order preserves positional identity, constructs offsets, rebases
/// correlated outputs, or is otherwise part of the result structure.
///
/// @tparam Range The input range type.
/// @tparam Result The result type.
/// @tparam LocalResult The per-block result type.
/// @tparam F0 Block task function type.
/// @tparam F1 Aggregation function type.
/// @param data Input data range.
/// @param result Output result (modified in-place).
/// @param local_result Initial value for per-block results.
/// @param task Function to process each block.
/// @param aggregate Function to combine results in order.
/// @param n_blocks Number of parallel blocks.
template <typename Range, typename Result, typename LocalResult, typename F0,
          typename F1>
auto blocked_reduce_sequenced_aggregate(
    const Range &data, Result &&result, LocalResult local_result, F0 task,
    F1 aggregate,
    std::size_t n_blocks = std::thread::hardware_concurrency() * 5) {
  if (data.size() == 0)
    return;
  if (std::thread::hardware_concurrency() == 1)
    n_blocks = 1;
  if (n_blocks == 1) {
    task(tf::make_range(data), local_result);
    aggregate(local_result, result);
    return;
  }
  const auto step = core::blocked_sequenced_chunk_stride(data.size(), n_blocks);
  auto n_tasks = (data.size() + step - 1) / step;

  core::std_vector<core::cache_aligned_slot<std::optional<LocalResult>>> locals(
      static_cast<std::size_t>(n_tasks));
  core::reduce_call_site<Range, std::remove_reference_t<Result>, LocalResult,
                         F0, F1>
      call_site{&data, std::size_t(step), &local_result, locals.data(),
                &task, &aggregate,        &result};
  core::run_sequenced_reduce_graph(&call_site, call_site.bodies(),
                                   static_cast<std::size_t>(n_tasks));
}

/// @ingroup core_algorithms
/// @brief Parallel reduction with ordered aggregation (inferred local result).
/// @overload
template <typename Range, typename Result, typename F0, typename F1>
auto blocked_reduce_sequenced_aggregate(
    const Range &data, Result &&result, F0 task, F1 aggregate,
    std::size_t n_blocks = std::thread::hardware_concurrency() * 5) {
  auto local_result = result;
  return blocked_reduce_sequenced_aggregate(
      data, static_cast<Result &&>(result), local_result, std::move(task),
      std::move(aggregate), n_blocks);
}

/// @ingroup core_algorithms
/// @brief Checked ordered reduction: one block below the parallel entry cost.
///
/// The serial path is the single-block path — the task sees the whole range
/// and the aggregate runs once — which is the input order the sequencer
/// exists to produce, so the result structure is unchanged.
template <typename Range, typename Result, typename LocalResult, typename F0,
          typename F1>
auto blocked_reduce_sequenced_aggregate(const Range &data, Result &&result,
                                        LocalResult local_result, F0 task,
                                        F1 aggregate, tf::checked_t c) {
  if (std::size_t(data.size()) < c.serial_below)
    return blocked_reduce_sequenced_aggregate(
        data, static_cast<Result &&>(result), std::move(local_result),
        std::move(task), std::move(aggregate), std::size_t(1));
  return blocked_reduce_sequenced_aggregate(
      data, static_cast<Result &&>(result), std::move(local_result),
      std::move(task), std::move(aggregate));
}

/// @ingroup core_algorithms
/// @brief Checked ordered reduction (inferred local result).
/// @overload
template <typename Range, typename Result, typename F0, typename F1>
auto blocked_reduce_sequenced_aggregate(const Range &data, Result &&result,
                                        F0 task, F1 aggregate,
                                        tf::checked_t c) {
  auto local_result = result;
  return blocked_reduce_sequenced_aggregate(
      data, static_cast<Result &&>(result), local_result, std::move(task),
      std::move(aggregate), c);
}
} // namespace tf
