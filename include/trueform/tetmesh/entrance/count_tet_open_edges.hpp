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
#include "../../core/algorithm/reduce.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>
#include <type_traits>

namespace tf::tetmesh {

/// What the edges of a triangle set leave open: `unbalanced`, the edges not
/// run as often one way as the other — zero exactly when the triangles
/// close a consistently wound surface — and among them `boundary`, the
/// edges one triangle alone holds: a free rim, which no winding closes.
/// Whether such a rim is a hole or a flap hanging at a non-manifold
/// attachment is @ref tf::has_holes' fact, not an edge's.
struct tet_open_edges {
  std::size_t unbalanced = 0;
  std::size_t boundary = 0;
};

/// One run of an edge: its corners ascending and `+1` or `-1` by which way
/// the triangle runs it.
template <typename Index> struct tet_edge_step {
  std::array<Index, 2> key;
  int step;
};

/// The edges of `triangles` left open (@ref tet_open_edges), counted on the
/// `records` the caller hands in. An edge from a corner to itself counts for
/// neither side.
template <typename Triangles, typename Index>
auto count_tet_open_edges(const Triangles &triangles,
                          tf::buffer<tet_edge_step<Index>> &records)
    -> tet_open_edges {
  using record_t = tet_edge_step<Index>;
  records.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(triangles.size()), records,
      [&triangles](std::size_t t, auto &out) {
        const auto &triangle = triangles[t];
        for (std::size_t k = 0; k < 3; ++k) {
          const Index a = Index(triangle[k]),
                      b = Index(triangle[k == 2 ? 0 : k + 1]);
          if (a != b)
            out.push_back({{std::min(a, b), std::max(a, b)}, a < b ? 1 : -1});
        }
      },
      tf::checked);
  tbb::parallel_sort(
      records.begin(), records.end(),
      [](const record_t &x, const record_t &y) { return x.key < y.key; });
  // each run is read from its first record
  return tf::reduce(
      tf::make_sequence_range(records.size()),
      [&records](tet_open_edges open, const auto &at) {
        if constexpr (std::is_same_v<std::decay_t<decltype(at)>,
                                     tet_open_edges>) {
          open.unbalanced += at.unbalanced;
          open.boundary += at.boundary;
        } else {
          const std::size_t r = std::size_t(at);
          if (r && records[r - 1].key == records[r].key)
            return open;
          std::size_t end = r;
          int balance = 0;
          for (; end < records.size() && records[end].key == records[r].key;
               ++end)
            balance += records[end].step;
          open.unbalanced += balance != 0;
          open.boundary += end - r == 1;
        }
        return open;
      },
      tet_open_edges{}, tf::checked);
}

} // namespace tf::tetmesh
