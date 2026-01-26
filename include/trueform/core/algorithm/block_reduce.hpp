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
#include "../range.hpp"
#include "tbb/flow_graph.h"
#include <thread>
namespace tf {

/// @ingroup core_algorithms
/// @brief Parallel blocked reduction with custom local and global aggregation.
///
/// Divides the input range into blocks, processes each block in parallel
/// using the `task` function to accumulate into a local result, then
/// aggregates all local results into the global result using `aggregate`.
///
/// The aggregation happens sequentially to ensure thread-safe access
/// to the global result.
///
/// @tparam Range The input range type.
/// @tparam Result The global result type.
/// @tparam LocalResult The thread-local result type.
/// @tparam F0 Block processing function: `void(block_range, LocalResult&)`.
/// @tparam F1 Aggregation function: `void(const LocalResult&, Result&)`.
/// @param data The input range to reduce.
/// @param result The global result (accumulated into).
/// @param local_result Template for thread-local storage.
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

  /*
   * We construct a dependecy graph
   *
   * work_0    work_1 ...    work_n
   *   |         |             |
   *   V         V             V
   * aggr_0 -> aggr_1 ... -> aggr_n
   */
  using msg_t = tbb::flow::continue_msg;
  using work_node_t = tbb::flow::function_node<int, LocalResult>;
  using aggregation_node_t = tbb::flow::function_node<LocalResult, msg_t>;
  tbb::flow::graph g{};

  aggregation_node_t aggregate_node{
      g, tbb::flow::serial,
      [&result, &aggregate](const LocalResult &local_result) {
        auto aggregate_f = aggregate;
        aggregate_f(local_result, result);
      }};
  work_node_t work_node{
      g, tbb::flow::unlimited, [&data, &local_result, step, task](int i) {
        auto r = tf::make_range(
            data.begin() + i * step,
            data.begin() +
                std::min(decltype(data.size())((i + 1) * step), data.size()));
        // we copy here because a copy in the lambda member
        // would be const
        auto local_resultt = local_result;
        task(r, local_resultt);
        return local_resultt;
      }};

  tbb::flow::make_edge(work_node, aggregate_node);

  for (decltype(n_tasks) i = 0; i < n_tasks; ++i)
    work_node.try_put(int(i));
  g.wait_for_all();
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
} // namespace tf
