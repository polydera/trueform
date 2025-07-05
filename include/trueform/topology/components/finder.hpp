/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/parallel_apply.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_map.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/views/constant.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/zip.hpp"
#include "tbb/flow_graph.h"
#include "tbb/task_group.h"
#include <atomic>

namespace tf::topology {
template <typename Index> class connected_components_finder {
  using label_t = short;

public:
  template <typename Range0, typename Range1, typename F>
  auto run(Range0 &&labels, const Range1 &mask, const F &applier) -> Index {
    clear();
    initialize(labels.size());
    auto n_labels = run_propagation(mask, applier);
    auto n_components = make_label_map(n_labels);
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
    _parent.clear();
    _root_map.clear();
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
    tf::buffer<Index> ids;
    ids.reserve(mask.size());
    for (auto [i, e] : tf::enumerate(mask)) {
      if (e)
        ids.push_back(i);
    }
    Index initial_size = ids.size();

    tbb::task_group walk_tasks;
    std::atomic<Index> n_processed{0};

    tbb::flow::graph g;
    tbb::flow::function_node<tf::hash_set<label_t>> collision_resolver(
        g, tbb::flow::serial, [&](const tf::hash_set<label_t> &collisions) {
          resolve_collisions(collisions);
        });

    auto walker_f = [&](std::pair<Index, label_t> initial) {
      tf::buffer<Index> stack;
      stack.reserve(200);
      stack.push_back(initial.first);
      tf::hash_set<label_t> collisions;
      auto processed =
          propagate_label(stack, collisions, initial.second, applier);
      if (processed) {
        n_processed.fetch_add(processed, std::memory_order_relaxed);
        collisions.insert(initial.second);
        collision_resolver.try_put(std::move(collisions));
      }
    };

    Index n_tasks = tbb::this_task_arena::max_concurrency() * 5;
    label_t current_label = 0;
    while (ids.size()) {
      Index step = std::ceil(float(ids.size()) / n_tasks);
      for (Index offset = 0; offset < Index(ids.size()); offset += step) {
        walk_tasks.run([&, id = ids[offset], label = current_label++] {
          walker_f(std::make_pair(id, label));
        });
      }
      walk_tasks.wait();
      if (n_processed.load(std::memory_order_relaxed) == initial_size)
        break;
      ids.erase(std::remove_if(ids.begin(), ids.end(),
                               [&](const auto &x) {
                                 return _work_labels[x].load(
                                            std::memory_order_relaxed) !=
                                        label_t{-1};
                               }),
                ids.end());
    }
    g.wait_for_all();
    return current_label;
  }

  auto find_parent(label_t x) -> label_t {
    label_t root = x;

    // Find root
    while (_parent.find(root) != _parent.end() && _parent[root] != root) {
      root = _parent[root];
    }

    // Path compression
    label_t current = x;
    while (_parent.find(current) != _parent.end() && _parent[current] != root) {
      label_t next = _parent[current];
      _parent[current] = root;
      current = next;
    }

    // If x was unseen, initialize its parent
    if (_parent.find(x) == _parent.end()) {
      _parent[x] = root;
    }

    return root;
  }

  auto resolve_collisions(const tf::hash_set<label_t> &collisions) {
    auto unite = [&](label_t a, label_t b) {
      label_t root_a = find_parent(a);
      label_t root_b = find_parent(b);
      if (root_a != root_b)
        _parent[root_b] = root_a;
    };

    if (collisions.empty())
      return;

    auto it = collisions.begin();
    label_t first = *it;
    label_t root_first = find_parent(first);

    for (++it; it != collisions.end(); ++it) {
      unite(root_first, *it);
    }
  }

  auto make_label_map(label_t n_labels) -> label_t {
    label_t _current_label = 0;
    _label_map.allocate(n_labels);
    _root_map.reserve(_parent.size());
    for (const auto &[label, _] : _parent) {
      // Repeated find with path compression
      label_t root = find_parent(label);
      auto iter = _root_map.find(root);
      if (iter == _root_map.end()) {
        _root_map[root] = _current_label;
        _label_map[label] = _current_label;
        ++_current_label;
      } else
        _label_map[label] = iter->second;
    }
    return _current_label;
  }

  auto initialize(std::size_t size) {
    _work_labels.allocate(size);
    tf::parallel_apply(_work_labels,
                       [](auto &x) { x.store(-1, std::memory_order_relaxed); });
  }

  tf::buffer<std::atomic<label_t>> _work_labels;
  tf::hash_map<label_t, label_t> _parent;
  tf::buffer<label_t> _label_map;
  tf::hash_map<label_t, label_t> _root_map;
};
} // namespace tf::topology
