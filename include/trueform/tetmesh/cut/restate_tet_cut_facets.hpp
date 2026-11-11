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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "./tet_changed_triangle.hpp"
#include "./tet_drain_cut.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// Every facet one of whose triangles `changed` is retired into `retired`,
/// ascending, and restated whole under its ticket: its kept triangles and
/// the pieces of its changed ones into `rows`, each with its input face in
/// `sources`, and `beside` naming, once each and ascending, every point a
/// piece holds that no site stands for (`seated`) with a corner of a
/// triangle it was made on — a point of an earlier drain no piece held then
/// as much as one this drain minted. The facets
/// retired are counted in `drain` as holding a traced triangle (`cut`) or
/// only one a split beyond reached. O(changed log changed + cut log cut +
/// Σ retired facets' triangles × one log).
template <typename Index, typename FacetOf, typename FacetOffsets, typename Cut,
          typename Parents, typename Triangles, typename CornersOf,
          typename Seated>
auto restate_tet_cut_facets(
    const tf::buffer<tet_changed_triangle<Index>> &changed,
    const tf::buffer<std::array<Index, 3>> &pieces, const Cut &cut,
    const FacetOf &facet_of, const FacetOffsets &facet_offsets,
    const Parents &parents, const Triangles &triangles,
    const CornersOf &corners_of, const Seated &seated,
    tf::buffer<Index> &retired, tf::blocked_buffer<Index, 3> &rows,
    tf::buffer<Index> &sources, tf::buffer<std::array<Index, 2>> &beside,
    tf::buffer<Index> &cut_facets, tet_drain_cut &drain) -> void {
  retired.clear();
  rows.clear();
  sources.clear();
  for (const auto &c : changed)
    retired.push_back(facet_of(c.triangle));
  std::sort(retired.begin(), retired.end());
  retired.erase_till_end(std::unique(retired.begin(), retired.end()));
  cut_facets.clear();
  for (auto t : cut)
    cut_facets.push_back(facet_of(t));
  std::sort(cut_facets.begin(), cut_facets.end());
  cut_facets.erase_till_end(std::unique(cut_facets.begin(), cut_facets.end()));
  for (auto f : retired) {
    const bool crossing =
        std::binary_search(cut_facets.begin(), cut_facets.end(), f);
    drain.retired_crossing += crossing;
    drain.retired_neighbour += !crossing;
  }
  beside.clear();
  for (auto f : retired) {
    const auto block = std::size_t(f);
    for (Index t = facet_offsets[block]; t < facet_offsets[block + 1]; ++t) {
      const Index source = Index(parents[std::size_t(t)]);
      const auto found =
          std::lower_bound(changed.begin(), changed.end(), t,
                           [](const tet_changed_triangle<Index> &c, Index key) {
                             return c.triangle < key;
                           });
      if (found == changed.end() || found->triangle != t) {
        rows.push_back(corners_of(t));
        sources.push_back(source);
        continue;
      }
      for (Index i = found->first; i < found->last; ++i) {
        rows.push_back(pieces[std::size_t(i)]);
        sources.push_back(source);
        for (auto v : pieces[std::size_t(i)])
          if (!seated(v))
            beside.push_back({v, Index(triangles[std::size_t(t)][0])});
      }
    }
  }
  // one corner per point: the first in row order
  std::stable_sort(beside.begin(), beside.end(),
                   [](const std::array<Index, 2> &a,
                      const std::array<Index, 2> &b) { return a[0] < b[0]; });
  beside.erase_till_end(
      std::unique(beside.begin(), beside.end(),
                  [](const std::array<Index, 2> &a,
                     const std::array<Index, 2> &b) { return a[0] == b[0]; }));
}

} // namespace tf::tetmesh
