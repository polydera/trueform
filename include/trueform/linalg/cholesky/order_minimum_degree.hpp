/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/buffer.hpp"
#include "./minimum_degree_scratch.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::linalg {

/// A fill-reducing elimination order of a symmetric graph: `order` names the
/// node eliminated at each position.
///
/// The approximate minimum degree walk over the quotient graph: each step
/// eliminates a variable of least approximate external degree, forms its
/// clique as a new element absorbing the elements it touched, and restates
/// the degrees of the clique's variables from `|Le \ Lp|`, computed for every
/// element at once against one watermark. An element its clique covers is
/// absorbed, a variable the clique alone touches is eliminated with the pivot,
/// and variables of identical lists merge into one supervariable, found by a
/// hash of their lists. A row denser than `10 sqrt(n)` waits until the end.
///
/// The graph is read only; it must name every edge from both ends, without
/// loops, as @ref tf::linalg::make_symmetric_adjacency states it.
template <typename Index>
auto order_minimum_degree(const tf::buffer<Index> &offsets,
                          const tf::buffer<Index> &adjacency,
                          tf::linalg::minimum_degree_scratch<Index> &s,
                          tf::buffer<Index> &order) -> void {
  using kind_t = tf::linalg::minimum_degree_kind;
  const std::size_t n = offsets.size() - 1;
  order.allocate(n);
  if (!n)
    return;
  const std::size_t dense =
      std::max(std::size_t(16), std::size_t(10.0 * std::sqrt(double(n))));

  s.kind.allocate(n);
  s.start.allocate(n);
  s.length.allocate(n);
  s.elements.allocate(n);
  s.weight.allocate(n);
  s.degree.allocate(n);
  s.next.allocate(n);
  s.last.allocate(n);
  s.hash_next.allocate(n);
  s.hash_of.allocate(n);
  s.representative.allocate(n);
  s.step_of.allocate(n);
  s.head.allocate_and_initialize(n + 1, Index(-1));
  s.hash_head.allocate_and_initialize(n, Index(-1));
  s.mark.allocate_and_initialize(n, std::size_t(0));

  for (std::size_t i = 0; i < n; ++i)
    s.kind[i] = std::size_t(offsets[i + 1] - offsets[i]) > dense
                    ? kind_t::deferred
                    : kind_t::variable;
  const std::size_t entries = std::size_t(offsets[n]);
  s.lists.allocate(entries + entries / 5 + 2 * n + 1);
  std::size_t free = 0;
  std::size_t live = 0;
  for (std::size_t i = 0; i < n; ++i) {
    s.start[i] = Index(free);
    s.elements[i] = Index(0);
    s.representative[i] = Index(-1);
    if (s.kind[i] == kind_t::variable) {
      for (Index at = offsets[i]; at < offsets[i + 1]; ++at) {
        const Index j = adjacency[std::size_t(at)];
        if (s.kind[std::size_t(j)] == kind_t::variable)
          s.lists[free++] = j;
      }
      ++live;
    }
    s.length[i] = Index(free - std::size_t(s.start[i]));
    s.weight[i] = Index(s.kind[i] == kind_t::variable ? 1 : 0);
    s.degree[i] = s.length[i];
  }

  const auto remove = [&s](Index i) {
    const std::size_t at = std::size_t(i);
    if (s.last[at] != Index(-1))
      s.next[std::size_t(s.last[at])] = s.next[at];
    else
      s.head[std::size_t(s.degree[at])] = s.next[at];
    if (s.next[at] != Index(-1))
      s.last[std::size_t(s.next[at])] = s.last[at];
  };
  const auto insert = [&s](Index i) {
    const std::size_t at = std::size_t(i);
    const std::size_t d = std::size_t(s.degree[at]);
    s.next[at] = s.head[d];
    s.last[at] = Index(-1);
    if (s.head[d] != Index(-1))
      s.last[std::size_t(s.head[d])] = i;
    s.head[d] = i;
  };
  // Live lists slide down over the dead ones, in the order they stand.
  const auto compact = [&s, n, &free]() {
    s.live.clear();
    for (std::size_t i = 0; i < n; ++i)
      if (s.kind[i] == kind_t::variable || s.kind[i] == kind_t::element)
        s.live.push_back(Index(i));
    std::sort(s.live.begin(), s.live.end(), [&s](Index a, Index b) {
      return s.start[std::size_t(a)] < s.start[std::size_t(b)];
    });
    std::size_t at = 0;
    for (auto i : s.live) {
      const std::size_t from = std::size_t(s.start[std::size_t(i)]);
      const std::size_t count = std::size_t(s.length[std::size_t(i)]);
      for (std::size_t k = 0; k < count; ++k)
        s.lists[at + k] = s.lists[from + k];
      s.start[std::size_t(i)] = Index(at);
      at += count;
    }
    free = at;
  };

  for (std::size_t i = 0; i < n; ++i)
    if (s.kind[i] == kind_t::variable)
      insert(Index(i));

  std::size_t watermark = 1;
  std::size_t eliminated = 0;
  std::size_t least = 0;
  s.pivots.clear();
  while (eliminated < live) {
    while (s.head[least] == Index(-1))
      ++least;
    const Index p = s.head[least];
    const std::size_t pivot = std::size_t(p);
    remove(p);
    s.pivots.push_back(p);
    std::size_t pivot_weight = std::size_t(s.weight[pivot]);
    eliminated += pivot_weight;
    s.weight[pivot] = Index(-s.weight[pivot]);

    std::size_t bound = std::size_t(s.length[pivot] - s.elements[pivot]);
    for (Index k = 0; k < s.elements[pivot]; ++k)
      bound += std::size_t(
          s.length[std::size_t(s.lists[std::size_t(s.start[pivot] + k)])]);
    if (free + bound > s.lists.size()) {
      compact();
      if (free + bound > s.lists.size())
        s.lists.reallocate(free + bound + s.lists.size() / 2);
    }

    // The clique: every variable the pivot reaches through an element or
    // directly, each once, flagged by a negated weight.
    const std::size_t clique = free;
    std::size_t clique_weight = 0;
    const auto take = [&](Index i) {
      const std::size_t at = std::size_t(i);
      if (s.kind[at] != kind_t::variable || s.weight[at] <= 0)
        return;
      clique_weight += std::size_t(s.weight[at]);
      s.weight[at] = Index(-s.weight[at]);
      s.lists[free++] = i;
      remove(i);
    };
    const std::size_t pivot_first = std::size_t(s.start[pivot]);
    for (Index k = 0; k < s.elements[pivot]; ++k) {
      const std::size_t e = std::size_t(s.lists[pivot_first + std::size_t(k)]);
      if (s.kind[e] != kind_t::element)
        continue;
      for (Index at = s.start[e]; at < s.start[e] + s.length[e]; ++at)
        take(s.lists[std::size_t(at)]);
      s.kind[e] = kind_t::absorbed;
    }
    for (Index k = s.elements[pivot]; k < s.length[pivot]; ++k)
      take(s.lists[pivot_first + std::size_t(k)]);
    s.kind[pivot] = kind_t::element;
    s.start[pivot] = Index(clique);
    s.length[pivot] = Index(free - clique);
    s.elements[pivot] = Index(0);

    // `mark[e] - watermark` becomes `|Le \ Lp|` for every element a clique
    // variable touches: first seen, it is `|Le|` less that variable's weight.
    for (std::size_t q = clique; q < free; ++q) {
      const std::size_t i = std::size_t(s.lists[q]);
      const std::size_t weight = std::size_t(-s.weight[i]);
      const std::size_t first = std::size_t(s.start[i]);
      for (Index k = 0; k < s.elements[i]; ++k) {
        const std::size_t e = std::size_t(s.lists[first + std::size_t(k)]);
        if (s.kind[e] != kind_t::element)
          continue;
        if (s.mark[e] >= watermark)
          s.mark[e] -= weight;
        else
          s.mark[e] = std::size_t(s.degree[e]) + watermark - weight;
      }
    }

    for (std::size_t q = clique; q < free; ++q) {
      const Index i = s.lists[q];
      const std::size_t at = std::size_t(i);
      const std::size_t weight = std::size_t(-s.weight[at]);
      const std::size_t first = std::size_t(s.start[at]);
      std::size_t write = first;
      std::size_t external = 0;
      std::size_t hash = 0;
      for (Index k = 0; k < s.elements[at]; ++k) {
        const Index e = s.lists[first + std::size_t(k)];
        if (s.kind[std::size_t(e)] != kind_t::element)
          continue;
        const std::size_t outside = s.mark[std::size_t(e)] - watermark;
        if (outside) {
          external += outside;
          s.lists[write++] = e;
          hash += std::size_t(e);
        } else
          s.kind[std::size_t(e)] = kind_t::absorbed;
      }
      const std::size_t kept_elements = write - first;
      for (Index k = s.elements[at]; k < s.length[at]; ++k) {
        const Index j = s.lists[first + std::size_t(k)];
        if (s.kind[std::size_t(j)] != kind_t::variable ||
            s.weight[std::size_t(j)] <= 0)
          continue;
        external += std::size_t(s.weight[std::size_t(j)]);
        s.lists[write++] = j;
        hash += std::size_t(j);
      }
      const std::size_t kept_variables = write - first - kept_elements;
      if (!kept_elements && !kept_variables) {
        s.kind[at] = kind_t::merged;
        s.representative[at] = p;
        s.weight[at] = Index(0);
        clique_weight -= weight;
        pivot_weight += weight;
        eliminated += weight;
        continue;
      }
      // The pivot dropped at least one entry from this list, so the pivot's
      // element takes the list's first slot and the slot it displaced
      // moves to the end.
      if (kept_variables)
        s.lists[write] = s.lists[first + kept_elements];
      if (kept_elements)
        s.lists[first + kept_elements] = s.lists[first];
      s.lists[first] = p;
      s.elements[at] = Index(kept_elements + 1);
      s.length[at] = Index(kept_elements + kept_variables + 1);
      s.degree[at] = Index(std::min(std::size_t(s.degree[at]), external));
      const std::size_t bucket = hash % n;
      s.hash_of[at] = Index(bucket);
      s.hash_next[at] = s.hash_head[bucket];
      s.hash_head[bucket] = i;
    }
    s.degree[pivot] = Index(clique_weight);
    watermark += n + 1;

    for (std::size_t q = clique; q < free; ++q) {
      const std::size_t i = std::size_t(s.lists[q]);
      if (s.kind[i] != kind_t::variable)
        continue;
      const std::size_t bucket = std::size_t(s.hash_of[i]);
      const Index chain = s.hash_head[bucket];
      if (chain == Index(-1))
        continue;
      s.hash_head[bucket] = Index(-1);
      for (Index a = chain; a != Index(-1); a = s.hash_next[std::size_t(a)]) {
        const std::size_t x = std::size_t(a);
        if (s.kind[x] != kind_t::variable)
          continue;
        for (Index k = 0; k < s.length[x]; ++k)
          s.mark[std::size_t(s.lists[std::size_t(s.start[x] + k)])] = watermark;
        Index previous = a;
        for (Index b = s.hash_next[x]; b != Index(-1);) {
          const std::size_t y = std::size_t(b);
          const Index after = s.hash_next[y];
          bool same = s.kind[y] == kind_t::variable &&
                      s.length[y] == s.length[x] &&
                      s.elements[y] == s.elements[x];
          for (Index k = 0; same && k < s.length[y]; ++k)
            same = s.mark[std::size_t(s.lists[std::size_t(s.start[y] + k)])] ==
                   watermark;
          if (same) {
            s.weight[x] = Index(s.weight[x] + s.weight[y]);
            s.weight[y] = Index(0);
            s.kind[y] = kind_t::merged;
            s.representative[y] = a;
            s.hash_next[std::size_t(previous)] = after;
          } else
            previous = b;
          b = after;
        }
        ++watermark;
      }
    }

    std::size_t kept = clique;
    for (std::size_t q = clique; q < free; ++q) {
      const Index i = s.lists[q];
      const std::size_t at = std::size_t(i);
      if (s.kind[at] != kind_t::variable)
        continue;
      const std::size_t weight = std::size_t(-s.weight[at]);
      s.weight[at] = Index(weight);
      const std::size_t degree =
          std::min(std::size_t(s.degree[at]) + clique_weight - weight,
                   live - eliminated - weight);
      s.degree[at] = Index(degree);
      insert(i);
      least = std::min(least, degree);
      s.lists[kept++] = i;
    }
    s.length[pivot] = Index(kept - clique);
    s.weight[pivot] = Index(pivot_weight);
    if (kept == clique)
      s.kind[pivot] = kind_t::absorbed;
    free = kept;
  }

  // A member follows its representatives to the pivot it is eliminated
  // with; members of one pivot stand together, in the pivots' order, and
  // the deferred rows follow every pivot.
  const std::size_t steps = s.pivots.size();
  for (std::size_t k = 0; k < steps; ++k)
    s.step_of[std::size_t(s.pivots[k])] = Index(k);
  s.pivot_offsets.allocate_and_initialize(steps + 1, Index(0));
  std::size_t deferred = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (s.kind[i] == kind_t::deferred) {
      ++deferred;
      continue;
    }
    std::size_t root = i;
    while (s.kind[root] == kind_t::merged)
      root = std::size_t(s.representative[root]);
    for (std::size_t at = i; s.kind[at] == kind_t::merged;) {
      const std::size_t up = std::size_t(s.representative[at]);
      s.representative[at] = Index(root);
      at = up;
    }
    ++s.pivot_offsets[std::size_t(s.step_of[root]) + 1];
  }
  for (std::size_t k = 0; k < steps; ++k)
    s.pivot_offsets[k + 1] = Index(s.pivot_offsets[k + 1] + s.pivot_offsets[k]);
  std::size_t tail = n - deferred;
  for (std::size_t i = 0; i < n; ++i) {
    if (s.kind[i] == kind_t::deferred) {
      order[tail++] = Index(i);
      continue;
    }
    const std::size_t root =
        s.kind[i] == kind_t::merged ? std::size_t(s.representative[i]) : i;
    auto &slot = s.pivot_offsets[std::size_t(s.step_of[root])];
    order[std::size_t(slot)] = Index(i);
    slot = Index(slot + 1);
  }
}

} // namespace tf::linalg
