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
#include "../../../../core/algorithm/generate_offset_blocks.hpp"
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/algorithm/sequenced_generate.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../manifold_edge_peer.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>
#include <tuple>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// State in `table` the protected `segments` with every edge bounding a
/// face of `faces` among them, each once with its slots ascending, in
/// `edges`, a block per face, the segments its rim runs along, and in
/// `holders`, aligned with `edges`, the face's own triangle on each, as the
/// edge it winds and its third corner. A face is a facet, coplanar
/// triangles wound alike; an edge is interior to it, and no segment,
/// exactly when it is shared by two of its triangles, run in opposite
/// directions, and by no other triangle of any face. A facet of one
/// triangle is bounded by its three edges. `link` is the triangles'
/// manifold edge link in the order `faces` blocks them, so a bounding edge
/// is read once, from the triangle its peer record names as its
/// representative.
template <typename Segments, typename Faces, typename Link, typename Index>
auto state_tet_facet_segments(const Segments &segments, const Faces &faces,
                              const Link &link,
                              tf::buffer<std::array<Index, 2>> &table,
                              tf::offset_block_buffer<Index, Index> &edges,
                              tf::buffer<std::array<Index, 3>> &holders)
    -> void {
  using key_t = std::array<Index, 2>;
  using peer_t = tf::manifold_edge_peer<Index>;
  const std::size_t given = segments.size(), n = faces.size();
  const auto edge = [](Index a, Index b) {
    return key_t{std::min(a, b), std::max(a, b)};
  };
  std::size_t n_triangles = 0;
  for (std::size_t f = 0; f < n; ++f)
    n_triangles += faces[f].size();
  tf::buffer<Index> facet_of;
  tf::buffer<Index> first;
  facet_of.allocate(n_triangles);
  first.allocate(n + 1);
  first[0] = 0;
  for (std::size_t f = 0; f < n; ++f)
    first[f + 1] = first[f] + Index(faces[f].size());
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&facet_of, &first](std::size_t f) {
        for (Index t = first[f]; t < first[f + 1]; ++t)
          facet_of[std::size_t(t)] = Index(f);
      },
      tf::checked);
  const auto triangle_of = [&faces, &first, &facet_of](std::size_t t) {
    const std::size_t f = std::size_t(facet_of[t]);
    return faces[f][t - std::size_t(first[f])];
  };
  const auto runs = [&triangle_of](std::size_t t, Index u, Index v) {
    const auto &c = triangle_of(t);
    for (std::size_t k = 0; k < 3; ++k)
      if (Index(c[k]) == u && Index(c[k == 2 ? 0 : k + 1]) == v)
        return true;
    return false;
  };
  const auto bounding = [&link, &facet_of, &triangle_of, &runs](std::size_t t,
                                                                std::size_t k) {
    const Index p = link[t][k].face_peer;
    if (p < 0)
      return true;
    if (facet_of[std::size_t(p)] != facet_of[t])
      return true;
    const auto &c = triangle_of(t);
    return runs(std::size_t(p), Index(c[k]), Index(c[k == 2 ? 0 : k + 1]));
  };
  const auto holds = [&link](std::size_t t, std::size_t k) {
    const Index p = link[t][k].face_peer;
    return p == peer_t::boundary || p == peer_t::non_manifold_representative ||
           (p >= 0 && std::size_t(p) > t);
  };
  table.allocate(given);
  tf::parallel_for_each(
      tf::make_sequence_range(given),
      [&table, &segments, &edge](std::size_t i) {
        table[i] = edge(Index(segments[i][0]), Index(segments[i][1]));
      },
      tf::checked);
  tf::sequenced_generate(
      tf::make_sequence_range(n_triangles), table,
      [&triangle_of, &bounding, &holds, &edge](std::size_t t, auto &out) {
        const auto &c = triangle_of(t);
        for (std::size_t k = 0; k < 3; ++k)
          if (holds(t, k) && bounding(t, k))
            out.push_back(edge(Index(c[k]), Index(c[k == 2 ? 0 : k + 1])));
      },
      tf::checked);
  tbb::parallel_sort(table.begin(), table.end());
  table.erase_till_end(std::unique(table.begin(), table.end()));
  std::tuple<tf::buffer<Index>, tf::buffer<std::array<Index, 3>>> rims;
  tf::generate_offset_blocks(
      tf::make_sequence_range(n), edges.offsets_buffer(), rims,
      [&faces, &first, &table, &edge, &bounding](std::size_t f, auto &out) {
        std::size_t t = std::size_t(first[f]);
        for (const auto &triangle : faces[f]) {
          for (std::size_t k = 0; k < 3; ++k)
            if (bounding(t, k)) {
              const Index a = Index(triangle[k]);
              const Index b = Index(triangle[k == 2 ? 0 : k + 1]);
              const Index w = Index(triangle[k == 0 ? 2 : k - 1]);
              std::get<0>(out).push_back(Index(
                  std::lower_bound(table.begin(), table.end(), edge(a, b)) -
                  table.begin()));
              std::get<1>(out).push_back({a, b, w});
            }
          ++t;
        }
      },
      tf::checked);
  edges.data_buffer() = std::move(std::get<0>(rims));
  holders = std::move(std::get<1>(rims));
}

} // namespace tf::topology::cdt::dt3
