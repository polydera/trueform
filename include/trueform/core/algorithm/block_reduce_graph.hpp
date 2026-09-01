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
#include "../range.hpp"
#include "tbb/flow_graph.h"
#include <algorithm>
#include <cstddef>
#include <optional>
namespace tf {
namespace core {

/// @brief The two bodies a blocked reduction hands its dependency graph: run
/// chunk `i` into its local slot, then aggregate that slot into the result.
///
/// TBB deep-copies a flow-graph message at every hop it makes — the work
/// node's return, the sequencer's buffer, the aggregation queue — so a message
/// carrying the local copies it several times per chunk. The message is the
/// chunk id; the local stays in the slot the work body grew it in.
struct reduce_bodies {
  void (*work)(void *state, int id);
  void (*consume)(void *state, int id);
};

/// @brief One blocked reduction, as its dependency graph sees it.
///
/// The chunk slots are the caller's: the work body grows chunk `i`'s local in
/// slot `i` from the local template, and the aggregation body consumes that
/// slot and releases it.
///
/// @tparam Range The input range type.
/// @tparam Result The result type.
/// @tparam LocalResult The per-chunk result type.
/// @tparam F0 Chunk task function type.
/// @tparam F1 Aggregation function type.
template <typename Range, typename Result, typename LocalResult, typename F0,
          typename F1>
struct reduce_call_site {
  const Range *data;
  std::size_t step;
  const LocalResult *local_result;
  cache_aligned_slot<std::optional<LocalResult>> *locals;
  const F0 *task;
  const F1 *aggregate;
  Result *result;

  static auto work(void *state, int i) -> void {
    auto &self = *static_cast<reduce_call_site *>(state);
    auto r = tf::make_range(
        self.data->begin() + i * self.step,
        self.data->begin() +
            std::min(decltype(self.data->size())((i + 1) * self.step),
                     self.data->size()));
    (*self.task)(r, self.locals[i].value.emplace(*self.local_result));
  }

  static auto consume(void *state, int i) -> void {
    auto &self = *static_cast<reduce_call_site *>(state);
    const LocalResult &local = *self.locals[i].value;
    auto aggregate_f = *self.aggregate;
    aggregate_f(local, *self.result);
    self.locals[i].value.reset();
  }

  auto bodies() const -> reduce_bodies { return {&work, &consume}; }
};

/// @brief Runs the chunks in parallel, aggregating in completion order.
///
/// @param state The @ref tf::core::reduce_call_site the bodies read.
/// @param bodies The call site's erased bodies.
/// @param n_tasks Number of chunks.
inline auto run_reduce_graph(void *state, reduce_bodies bodies,
                             std::size_t n_tasks) -> void {
  /*
   * We construct a dependency graph
   *
   * work_0    work_1 ...    work_n
   *   |         |             |
   *   V         V             V
   * aggr_0 -> aggr_1 ... -> aggr_n
   */
  using msg_t = tbb::flow::continue_msg;
  using work_node_t = tbb::flow::function_node<int, int>;
  using aggregation_node_t = tbb::flow::function_node<int, msg_t>;
  tbb::flow::graph g{};

  aggregation_node_t aggregate_node{
      g, tbb::flow::serial,
      [state, bodies](int i) { bodies.consume(state, i); }};
  work_node_t work_node{g, tbb::flow::unlimited, [state, bodies](int i) {
                          bodies.work(state, i);
                          return i;
                        }};

  tbb::flow::make_edge(work_node, aggregate_node);

  for (std::size_t i = 0; i < n_tasks; ++i)
    work_node.try_put(int(i));
  g.wait_for_all();
}

/// @brief Runs the chunks in parallel, aggregating in chunk order.
///
/// @param state The @ref tf::core::reduce_call_site the bodies read.
/// @param bodies The call site's erased bodies.
/// @param n_tasks Number of chunks.
inline auto run_sequenced_reduce_graph(void *state, reduce_bodies bodies,
                                       std::size_t n_tasks) -> void {
  /*
   * We construct a dependency graph
   *
   * work_0    work_1 ...    work_n
   *   |         |             |
   *   V         V             V
   * aggr_0 -> aggr_1 ... -> aggr_n
   */
  using msg_t = tbb::flow::continue_msg;
  using work_node_t = tbb::flow::function_node<int, int>;
  using aggregation_node_t = tbb::flow::function_node<int, msg_t>;
  using sequenced_t = tbb::flow::sequencer_node<int>;
  tbb::flow::graph g{};

  aggregation_node_t aggregate_node{
      g, tbb::flow::serial,
      [state, bodies](int i) { bodies.consume(state, i); }};
  work_node_t work_node{g, tbb::flow::unlimited, [state, bodies](int i) {
                          bodies.work(state, i);
                          return i;
                        }};
  sequenced_t sequencer{g, [](int i) { return std::size_t(i); }};

  tbb::flow::make_edge(work_node, sequencer);
  tbb::flow::make_edge(sequencer, aggregate_node);

  for (std::size_t i = 0; i < n_tasks; ++i)
    work_node.try_put(int(i));
  g.wait_for_all();
}

} // namespace core
} // namespace tf
