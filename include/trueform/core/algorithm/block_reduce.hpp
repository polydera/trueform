/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../range.hpp"
#include "tbb/flow_graph.h"
#include <thread>
namespace tf {
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

template <typename Range, typename Result, typename F0, typename F1>
auto blocked_reduce(const Range &data, Result &&result, F0 task, F1 aggregate,
                    std::size_t n_blocks = std::thread::hardware_concurrency() *
                                           5) {
  auto local_result = result;
  return blocked_reduce(data, static_cast<Result &&>(result), local_result,
                        std::move(task), std::move(aggregate), n_blocks);
}
} // namespace tf
