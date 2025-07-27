/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_apply.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/local_buffer.hpp"
#include "../../core/views/constant.hpp"
#include "../../core/views/zip.hpp"
#include "tbb/task_group.h"
#include <atomic>

namespace tf::topology {
template <typename Index, typename LabelType = short>
class connected_components_finder {
  using label_t = LabelType;

public:
  template <typename Range0, typename Range1, typename F>
  auto run(Range0 &&labels, const Range1 &mask, const F &applier) -> label_t {
    clear();
    initialize(labels.size());
    auto n_components = run_propagation(mask, applier);
    tf::parallel_apply(tf::zip(_work_labels, labels), [&](auto pair) {
      auto &&[wl, l] = pair;
      l = _label_map[wl.load(std::memory_order_relaxed)];
    });
    return n_components;
  }

  template <typename Range, typename F>
  auto run(Range &&labels, const F &applier) -> Index {
    return run(labels, tf::make_constant_range(true, labels.size()), applier);
  }

  auto clear() {
    _label_map.clear();
    _work_labels.clear();
  }

private:
  template <typename F>
  auto propagate_label(tf::buffer<Index> &stack,
                       tf::hash_set<label_t> &collisions, label_t label,
                       F applier) {
    Index count = 0;
    auto pusher = [&stack](Index id) { stack.push_back(id); };
    while (stack.size()) {
      Index current = stack.back();
      stack.pop_back();
      label_t expected{-1};
      if (!_work_labels[current].compare_exchange_strong(
              expected, label, std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        if (expected != label)
          collisions.insert(expected);
        continue;
      }
      ++count;
      applier(current, pusher);
    }
    return count;
  }

  template <typename Range, typename F>
  auto run_propagation(const Range &mask, const F &applier) -> label_t {
    std::atomic<label_t> current_label{0};
    auto n_tasks =
        std::max(Index(1), Index(tbb::this_task_arena::max_concurrency() - 1));
    if (n_tasks != 1)
      n_tasks *= 5;
    auto size = Index(mask.size());
    auto local_size = Index(std::ceil(float(size) / n_tasks));
    std::atomic<Index> n_left{
        Index(std::count(mask.begin(), mask.end(), true))};

    tbb::task_group walkers;
    tf::local_buffer<std::array<label_t, 2>> merge_labels;

    auto make_task_f = [&](Index task_i) {
      return [&, task_i] {
        tf::buffer<Index> stack;
        tf::hash_set<label_t> collisions;
        stack.reserve(200);
        for (Index i = local_size * task_i;
             i < std::min(local_size * (task_i + 1), size); ++i) {

          if (!mask[i] || _work_labels[i] != -1)
            continue;
          collisions.clear();
          auto label = current_label++;
          stack.push_back(i);
          auto processed = propagate_label(stack, collisions, label, applier);
          if (processed) {
            if (!collisions.size())
              merge_labels.push_back({label, label});
            for (auto m_label : collisions)
              merge_labels.push_back({label, m_label});
            n_left.fetch_sub(processed);
          }
        }
        if (!n_left.load()) {
          walkers.cancel();
          return;
        }
      };
    };

    for (Index i = 0; i < n_tasks; ++i)
      walkers.run(make_task_f(i));
    walkers.wait();
    _label_map.allocate(current_label.load(std::memory_order_relaxed));
    return tf::make_sparse_equivalence_class_map(merge_labels.to_buffer(),
                                                 _label_map);
  }

  auto initialize(std::size_t size) {
    _work_labels.allocate(size);
    tf::parallel_apply(_work_labels,
                       [](auto &x) { x.store(-1, std::memory_order_relaxed); });
  }

  tf::buffer<std::atomic<label_t>> _work_labels;
  tf::buffer<label_t> _label_map;
};
} // namespace tf::topology
