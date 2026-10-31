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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./for_each_ring_sector.hpp"
#include "./make_plane_radial_fans.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// The rings that state a self-sided component: per component `convicted`
/// marks, the rings whose sectors join its two sides — the sector pairs the
/// domain partition merges (@ref emit_domain_merges) walked from one side to
/// the other through a breadth-first forest of the side graph, so the rings
/// are the ones a path between the two sides crosses, not the component's
/// extent. Each row is `{component, fan}`, ascending and once; a component
/// whose two sides no ring's sectors join — only an open component's
/// self-merge does — names none.
template <typename CsgGraph>
auto make_self_sided_rings(const CsgGraph &g, const tf::buffer<char> &convicted)
    -> tf::buffer<std::array<typename CsgGraph::index_type, 2>> {
  using Index = typename CsgGraph::index_type;
  using edge_t = std::array<Index, 3>;
  tf::buffer<std::array<Index, 2>> out;
  const auto &d = g.descriptor();
  const auto &fans = d.fans;
  const auto tl = g.labels().triangle_labels();
  const auto eor = g.arrangement().exposed_of_row();
  const auto n_nodes = std::size_t(2) * convicted.size();

  // {node, other, fan}: every sector a certified ring merges, both ways
  tf::buffer<edge_t> edges;
  tf::generic_generate(
      tf::make_sequence_range(Index(fans.pieces.size())), edges,
      [&](Index fan, tf::buffer<edge_t> &local) {
        if (!d.valid[std::size_t(fan)] ||
            fans.verdicts[std::size_t(fan)] !=
                plane_radial_ring_verdict::certified)
          return;
        for_each_ring_sector(tl, fans, eor, fan,
                             [&](Index a, Index b, Index, Index) {
                               if (a == b)
                                 return;
                               local.push_back({a, b, fan});
                               local.push_back({b, a, fan});
                             });
      },
      tf::checked);
  tbb::parallel_sort(edges.begin(), edges.end());
  edges.erase_till_end(std::unique(edges.begin(), edges.end(),
                                   [](const edge_t &x, const edge_t &y) {
                                     return x[0] == y[0] && x[1] == y[1];
                                   }));
  tf::buffer<Index> offsets;
  offsets.allocate(n_nodes + 1);
  tf::parallel_transform(
      tf::make_sequence_range(n_nodes + 1), offsets,
      [&](std::size_t node) {
        return Index(
            std::lower_bound(edges.begin(), edges.end(),
                             edge_t{Index(node), Index(-1), Index(-1)}) -
            edges.begin());
      },
      tf::checked);

  // the forest grows from each convicted side not yet reached; a domain is
  // one tree, so both sides of a component stand in it
  const auto none = Index(-1);
  tf::buffer<Index> parent, parent_fan, depth, root;
  parent.allocate(n_nodes);
  root.allocate(n_nodes);
  parent_fan.allocate(n_nodes);
  depth.allocate(n_nodes);
  std::fill(depth.begin(), depth.end(), none);
  tf::buffer<Index> queue;
  for (std::size_t c = 0; c < convicted.size(); ++c) {
    if (!convicted[c] || depth[2 * c] != none)
      continue;
    queue.clear();
    queue.push_back(Index(2 * c));
    parent[2 * c] = none;
    parent_fan[2 * c] = none;
    depth[2 * c] = Index(0);
    root[2 * c] = Index(2 * c);
    for (std::size_t at = 0; at < queue.size(); ++at) {
      const auto u = queue[at];
      for (auto e = offsets[std::size_t(u)]; e < offsets[std::size_t(u) + 1];
           ++e) {
        const auto v = edges[std::size_t(e)][1];
        if (depth[std::size_t(v)] != none)
          continue;
        depth[std::size_t(v)] = depth[std::size_t(u)] + Index(1);
        parent[std::size_t(v)] = u;
        parent_fan[std::size_t(v)] = edges[std::size_t(e)][2];
        root[std::size_t(v)] = root[std::size_t(u)];
        queue.push_back(v);
      }
    }
  }
  for (std::size_t c = 0; c < convicted.size(); ++c) {
    if (!convicted[c] || depth[2 * c + 1] == none ||
        root[2 * c + 1] != root[2 * c])
      continue;
    auto u = Index(2 * c), v = Index(2 * c + 1);
    while (u != v) {
      if (depth[std::size_t(u)] < depth[std::size_t(v)])
        std::swap(u, v);
      out.push_back({Index(c), parent_fan[std::size_t(u)]});
      u = parent[std::size_t(u)];
    }
  }
  std::sort(out.begin(), out.end());
  out.erase_till_end(std::unique(out.begin(), out.end()));
  return out;
}

} // namespace tf::csg::graph
