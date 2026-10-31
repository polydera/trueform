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
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "./census_conviction.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace tf::csg::graph {

/// THE BLAST-RADIUS LAW: a round states only the fusions its own evidence
/// names, at the sites the evidence names; the component is never the
/// grain. A site is a set of triangles, and it states the near pairs (@ref
/// make_live_near_identities) between their corners. Each conviction names
/// its own: an odd ring the triangles of its pages; a self-sided component
/// every rounding crossing that reaches it, the crossing's two triangles,
/// and, where none does, the triangles of the pages of the rings that state
/// it (@ref make_self_sided_rings). A failed carrier holds no triangle and
/// names none, and a conviction whose evidence names no site states nothing:
/// the standing graph is its answer.
///
/// A pair is `{x, y, w_x, w_y}`: the two members' flat ids, `x < y`, and per
/// member the plane of a site triangle naming it, `-1` for an uncut face.
/// The pairs are ascending and once.
template <typename CsgGraph>
auto collect_census_id_pairs(
    const CsgGraph &g,
    const census_conviction<typename CsgGraph::index_type> &conviction,
    tf::buffer<std::array<typename CsgGraph::index_type, 4>> &pairs) -> void {
  using Index = typename CsgGraph::index_type;
  using corner_t = std::array<Index, 3>;
  using vsource = tf::intersect::graph::vertex_source;
  pairs.clear();
  const auto &near_identities = conviction.near_identities;
  if (near_identities.size() == 0)
    return;
  const auto &fans = g.descriptor().fans;
  const auto &ag = g.arrangement();
  const auto &arr = ag.arrangement();
  const auto triangles = arr.triangles();
  static_assert(std::is_same_v<std::decay_t<decltype(triangles[0][0])>, Index>,
                "pairs are stated in the triangles' corner type");
  const auto n_flat = arr.n_flat_points();
  const auto tris = ag.global().exposed_tris();
  const auto descriptors = ag.global().exposed_descriptors();
  const auto row_of = ag.row_of();
  const auto &vertex_offsets = ag.vertex_offsets();
  const auto apply_form = ag.apply_to_form();
  const auto n_exposed = Index(tris.size());
  const auto n_odd = Index(conviction.odd_rings.size());
  const auto n_components = Index(conviction.convicted.size());
  const auto ring_corners = [&](Index site, std::size_t f,
                                tf::buffer<corner_t> &out) {
    for (auto p = fans.page_offsets[f]; p < fans.page_offsets[f + 1]; ++p)
      for (auto r : fans.rows[std::size_t(p)]) {
        if (r < 0)
          continue;
        const auto t = r / 3;
        const auto plane = arr.plane_of_triangle(Index(t));
        for (const auto corner : triangles[std::size_t(t)])
          out.push_back({site, corner, plane});
      }
  };
  // {site, corner, witness}: an odd ring's pages, the rings stating a
  // self-sided component, then the crossings that reach one
  tf::buffer<corner_t> corners;
  tf::generic_generate(
      tf::make_sequence_range(std::size_t(n_odd)), corners,
      [&](std::size_t k, tf::buffer<corner_t> &out) {
        ring_corners(Index(k), std::size_t(conviction.odd_rings[k]), out);
      },
      tf::checked);
  tf::generic_generate(
      conviction.sided_rings, corners,
      [&](const std::array<Index, 2> &ring, tf::buffer<corner_t> &out) {
        ring_corners(n_odd + ring[0], std::size_t(ring[1]), out);
      },
      tf::checked);
  tf::generic_generate(
      conviction.reaching, corners,
      [&](Index k, tf::buffer<corner_t> &out) {
        const auto &record = conviction.rounding[std::size_t(k)];
        const auto site = n_odd + n_components + k;
        for (std::size_t s = 0; s < 2; ++s) {
          const auto t = record[s];
          if (t < n_exposed) {
            const auto plane = arr.plane_of_triangle(row_of[std::size_t(t)]);
            for (std::size_t c = 0; c < 3; ++c) {
              const auto &v = tris[std::size_t(t)][c];
              out.push_back(
                  {site, v.source == vsource::original ? v.id : n_flat + v.id,
                   plane});
            }
            continue;
          }
          const auto &descriptor = descriptors[std::size_t(t - n_exposed)];
          apply_form(Index(descriptor.tag), [&](const auto &form) {
            for (const auto v : form.faces()[descriptor.object])
              out.push_back(
                  {site, vertex_offsets[std::size_t(descriptor.tag)] + Index(v),
                   Index(-1)});
          });
        }
      },
      tf::checked);
  if (corners.size() == 0)
    return;
  tbb::parallel_sort(corners.begin(), corners.end());
  corners.erase_till_end(std::unique(corners.begin(), corners.end(),
                                     [](const corner_t &a, const corner_t &b) {
                                       return a[0] == b[0] && a[1] == b[1];
                                     }));
  tf::buffer<Index> site_offsets;
  site_offsets.reserve(corners.size() + 1);
  tf::compute_offsets(
      corners, std::back_inserter(site_offsets), Index(0),
      [](const corner_t &a, const corner_t &b) { return a[0] == b[0]; });
  tf::generic_generate(
      tf::make_offset_block_range(site_offsets, tf::make_range(corners)), pairs,
      [&near_identities](const auto &site,
                         tf::buffer<std::array<Index, 4>> &out) {
        const auto by_corner = [](const corner_t &c, Index id) {
          return c[1] < id;
        };
        for (const auto &x : site) {
          auto at =
              std::lower_bound(near_identities.begin(), near_identities.end(),
                               std::array<Index, 2>{x[1], Index(-1)});
          for (; at != near_identities.end() && (*at)[0] == x[1]; ++at) {
            const auto y =
                std::lower_bound(site.begin(), site.end(), (*at)[1], by_corner);
            if (y != site.end() && (*y)[1] == (*at)[1])
              out.push_back({x[1], (*y)[1], x[2], (*y)[2]});
          }
        }
      },
      tf::checked);
  tbb::parallel_sort(pairs.begin(), pairs.end());
  pairs.erase_till_end(std::unique(
      pairs.begin(), pairs.end(),
      [](const std::array<Index, 4> &a, const std::array<Index, 4> &b) {
        return a[0] == b[0] && a[1] == b[1];
      }));
}

} // namespace tf::csg::graph
