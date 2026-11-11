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
#include "../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/algorithm/parallel_iota.hpp"
#include "../../../../core/algorithm/sequenced_generate.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../../../../core/views/drop.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../manifold_edge_peer.hpp"
#include "./bucket_tet_records.hpp"
#include "./tet_facet_segment_scratch.hpp"
#include "./tet_site_segments.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// State in `table` the protected `segments` with every edge bounding a
/// face of `faces` among them, each once with its slots ascending, below
/// `n_sites`, and append to `edges` a block per face, the segments its rim
/// runs along, and in `holders`, aligned with `edges`, the face's own triangle
/// on each, as the edge it winds and its third corner. A face is a facet,
/// coplanar triangles wound alike; an edge is interior to it, and no
/// segment, exactly when it is shared by two of its triangles, run in
/// opposite directions, and by no other triangle of any face. A facet of
/// one triangle is bounded by its three edges. `link` is the triangles'
/// manifold edge link in the order `faces` blocks them, so a bounding edge
/// is stated once, from the triangle its peer record names as its
/// representative, and a rim names its segment through that
/// representative's position; only an edge three triangles share is
/// looked up among them. The statement is a diff on the tables it is handed:
/// `table` holds the segments already stated, `by_site` them per lower site
/// (@ref tet_site_segments); an edge a segment `live(s)` still names holds
/// keeps the first such id, and the edges none holds are appended in key
/// order and chained at their lower sites, so a first statement over empty
/// tables numbers the segments in key order. `scratch` is the caller's own
/// tables for the pass (@ref tet_facet_segment_scratch).
template <typename Segments, typename Faces, typename Link, typename Index,
          typename Live>
auto state_tet_facet_segments(const Segments &segments, const Faces &faces,
                              const Link &link, const Live &live,
                              std::size_t n_sites,
                              tf::buffer<std::array<Index, 2>> &table,
                              tet_site_segments<Index> &by_site,
                              tf::offset_block_buffer<Index, Index> &edges,
                              tf::buffer<std::array<Index, 3>> &holders,
                              tet_facet_segment_scratch<Index> &scratch)
    -> void {
  using key_t = std::array<Index, 2>;
  using peer_t = tf::manifold_edge_peer<Index>;
  using record_t = tet_facet_segment_record<Index>;
  const std::size_t given = segments.size(), n = faces.size();
  const auto edge = [](Index a, Index b) {
    return key_t{std::min(a, b), std::max(a, b)};
  };
  std::size_t n_triangles = 0;
  for (std::size_t f = 0; f < n; ++f)
    n_triangles += faces[f].size();
  auto &facet_of = scratch.facet_of;
  auto &first = scratch.first;
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
  auto &records = scratch.records;
  records.allocate(given);
  tf::parallel_for_each(
      tf::make_sequence_range(given),
      [&records, &segments, &edge](std::size_t i) {
        records[i] = {edge(Index(segments[i][0]), Index(segments[i][1])),
                      Index(-1)};
      },
      tf::checked);
  tf::sequenced_generate(
      tf::make_sequence_range(n_triangles), records,
      [&triangle_of, &bounding, &holds, &edge](std::size_t t, auto &out) {
        const auto &c = triangle_of(t);
        for (std::size_t k = 0; k < 3; ++k)
          if (holds(t, k) && bounding(t, k))
            out.push_back(
                record_t{edge(Index(c[k]), Index(c[k == 2 ? 0 : k + 1])),
                         Index(3 * t + k)});
      },
      tf::checked);
  bucket_tet_records(
      records, n_sites, scratch.runs, scratch.record_scratch, scratch.counts,
      [](const record_t &record) { return record.key[0]; },
      [](const record_t &x, const record_t &y) { return x.key[1] < y.key[1]; });
  auto &segment_at = scratch.segment_at;
  auto &keys = scratch.keys;
  segment_at.allocate(3 * n_triangles);
  keys.allocate(records.size());
  std::size_t m = 0;
  for (std::size_t i = 0; i < records.size(); ++i) {
    if (!m || keys[m - 1] != records[i].key)
      keys[m++] = records[i].key;
    if (records[i].at != Index(-1))
      segment_at[std::size_t(records[i].at)] = Index(m - 1);
  }
  keys.erase_till_end(keys.begin() + m);
  auto &global_of = scratch.global_of;
  global_of.allocate(m);
  auto &head = by_site.head;
  auto &tail = by_site.tail;
  auto &next = by_site.next;
  const std::size_t standing = table.size();
  const std::size_t chained = by_site.chained;
  if (standing > chained) {
    // the unchained segments are the last statement's, appended in key
    // order, so each lower site's run of them is contiguous: chained within
    // the run, its first joined to the site's standing tail, then its last
    // made the tail
    if constexpr (tf::core::verification)
      tf::core::verify(
          std::is_sorted(table.begin() + std::ptrdiff_t(chained), table.end()),
          "the unchained segments are one statement's run");
    if (head.size() < n_sites) {
      const std::size_t known = head.size();
      head.reallocate(n_sites);
      tail.reallocate(n_sites);
      tf::parallel_fill(tf::drop(head, known), Index(-1));
      tf::parallel_fill(tf::drop(tail, known), Index(-1));
    }
    next.reallocate(standing);
    const auto unchained = tf::make_sequence_range(chained, standing);
    tf::parallel_for_each(
        unchained,
        [&table, &head, &tail, &next, chained, standing](std::size_t s) {
          const auto a = table[s][0];
          next[s] = s + 1 < standing && table[s + 1][0] == a ? Index(s + 1)
                                                             : Index(-1);
          if (s > chained && table[s - 1][0] == a)
            return;
          if (tail[std::size_t(a)] == Index(-1))
            head[std::size_t(a)] = Index(s);
          else
            next[std::size_t(tail[std::size_t(a)])] = Index(s);
        },
        tf::checked);
    tf::parallel_for_each(
        unchained,
        [&table, &tail, &next](std::size_t s) {
          if (next[s] == Index(-1))
            tail[std::size_t(table[s][0])] = Index(s);
        },
        tf::checked);
    by_site.chained = standing;
  }
  std::size_t fresh = standing;
  if (!standing) {
    tf::parallel_iota(global_of, Index(0));
    fresh = m;
  } else {
    tf::parallel_for_each(
        tf::make_sequence_range(m),
        [&global_of, &keys, &table, &head, &next, &live](std::size_t i) {
          global_of[i] = Index(-1);
          const auto a = std::size_t(keys[i][0]);
          if (a >= head.size())
            return;
          for (Index s = head[a]; s != Index(-1); s = next[std::size_t(s)])
            if (table[std::size_t(s)] == keys[i] && live(s)) {
              global_of[i] = s;
              return;
            }
        },
        tf::checked);
    for (std::size_t i = 0; i < m; ++i)
      if (global_of[i] == Index(-1))
        global_of[i] = Index(fresh++);
  }
  table.reallocate(fresh);
  tf::parallel_for_each(
      tf::make_sequence_range(m),
      [&global_of, &keys, &table, standing](std::size_t i) {
        if (std::size_t(global_of[i]) >= standing)
          table[std::size_t(global_of[i])] = keys[i];
      },
      tf::checked);
  const auto local_of = [&link, &keys, &edge, &triangle_of, &holds,
                         &segment_at](std::size_t t, std::size_t k, Index a,
                                      Index b) {
    if (holds(t, k))
      return segment_at[3 * t + k];
    const Index p = link[t][k].face_peer;
    if (p >= 0) {
      const auto &c = triangle_of(std::size_t(p));
      for (std::size_t j = 0; j < 3; ++j)
        if (edge(Index(c[j]), Index(c[j == 2 ? 0 : j + 1])) == edge(a, b))
          return segment_at[3 * std::size_t(p) + j];
    }
    return Index(std::lower_bound(keys.begin(), keys.end(), edge(a, b)) -
                 keys.begin());
  };
  auto &offsets = edges.offsets_buffer();
  if (!offsets.size())
    offsets.push_back(Index(0));
  const std::size_t base = offsets.size() - 1;
  offsets.reallocate(base + n + 1);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&offsets, &first, &bounding, base](std::size_t f) {
        Index count = 0;
        for (Index t = first[f]; t < first[f + 1]; ++t)
          for (std::size_t k = 0; k < 3; ++k)
            count += Index(bounding(std::size_t(t), k));
        offsets[base + f + 1] = count;
      },
      tf::checked);
  for (std::size_t f = 0; f < n; ++f)
    offsets[base + f + 1] += offsets[base + f];
  auto &rims = edges.data_buffer();
  rims.reallocate(std::size_t(offsets[base + n]));
  holders.reallocate(std::size_t(offsets[base + n]));
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t f) {
        auto at = std::size_t(offsets[base + f]);
        std::size_t t = std::size_t(first[f]);
        for (const auto &triangle : faces[f]) {
          for (std::size_t k = 0; k < 3; ++k)
            if (bounding(t, k)) {
              const Index a = Index(triangle[k]);
              const Index b = Index(triangle[k == 2 ? 0 : k + 1]);
              const Index w = Index(triangle[k == 0 ? 2 : k - 1]);
              const Index s = global_of[std::size_t(local_of(t, k, a, b))];
              if constexpr (tf::core::verification)
                tf::core::verify(table[std::size_t(s)] == edge(a, b),
                                 "a rim names its segment through the "
                                 "representative of its edge");
              rims[at] = s;
              holders[at] = {a, b, w};
              ++at;
            }
          ++t;
        }
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3
