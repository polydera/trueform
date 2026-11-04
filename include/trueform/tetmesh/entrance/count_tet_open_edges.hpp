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
/// edges one triangle alone holds: a hole's rim, which no winding closes.
struct tet_open_edges {
  std::size_t unbalanced = 0;
  std::size_t boundary = 0;
};

/// The edges of `triangles` left open (@ref tet_open_edges). An edge from
/// a corner to itself counts for neither side.
template <typename Triangles>
auto count_tet_open_edges(const Triangles &triangles) -> tet_open_edges {
  using Index = std::decay_t<decltype(triangles[0][0])>;
  struct record_t {
    std::array<Index, 2> key;
    int step;
  };
  tf::buffer<record_t> records;
  tf::sequenced_generate(
      tf::make_sequence_range(triangles.size()), records,
      [&triangles](std::size_t t, auto &out) {
        const auto &triangle = triangles[t];
        for (std::size_t k = 0; k < 3; ++k) {
          const Index a = triangle[k], b = triangle[k == 2 ? 0 : k + 1];
          if (a != b)
            out.push_back({{std::min(a, b), std::max(a, b)}, a < b ? 1 : -1});
        }
      },
      tf::checked);
  tbb::parallel_sort(
      records.begin(), records.end(),
      [](const record_t &x, const record_t &y) { return x.key < y.key; });
  tet_open_edges open;
  for (std::size_t r = 0; r < records.size();) {
    std::size_t end = r;
    int balance = 0;
    for (; end < records.size() && records[end].key == records[r].key; ++end)
      balance += records[end].step;
    open.unbalanced += balance != 0;
    open.boundary += end - r == 1;
    r = end;
  }
  return open;
}

} // namespace tf::tetmesh
