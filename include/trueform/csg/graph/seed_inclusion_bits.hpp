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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/transformed.hpp"
#include "../../cut/arrangement_graph.hpp"
#include "../../cut/arrangements/arrangement_descriptor.hpp"
#include "../../cut/arrangements/compute_domain_inclusions.hpp"
#include "../../cut/face_cuts.hpp"
#include "../../exact/segment_hits_aabb.hpp"
#include "../../exact/triangle_segment_intersection.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../spatial/search.hpp"
#include "tbb/parallel_sort.h"
#include "./compute_bundle_aabbs.hpp"
#include "./winding_side.hpp"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Pick BFS seeds and raycast-seed per-bundle outer-env bits.
///
/// Each bundle's outer-shell is the most-negative-volume incident
/// domain (same formula as `make_nesting_merges`). The globally
/// most-negative outer-shell is `null_seed`, anchored at zero bits.
/// Other bundles are segment-cast to a far point against per-form
/// AABB trees (`tf::search` with exact `segment_hits_aabb` at every
/// BV node, exact `triangle_segment_intersect_point_sos` at leaves);
/// per-form parity goes into `inc.bits[outer_env(bi)]`.
///
/// Sheet forms (`is_sheet_tag`) are side-classified instead: the AABB
/// prefilter is skipped (a bundle is above or below a sheet even
/// without overlap) and the bit is the sign of the generalized winding
/// number (@ref tf::csg::graph::winding_side) — set when the seed is
/// behind the sheet's normal. One batched pass per sheet over its
/// triangles classifies all bundles needing it.
template <typename Forms, typename Index, typename Int, typename Real,
          std::size_t Dims, typename VolT>
auto seed_inclusion_bits(
    tf::cut::domain_inclusions &inc,
    const tf::cut::arrangement_descriptor<Index> &desc,
    const tf::arrangement_graph<Index> &ag,
    const tf::face_cuts<Index, Int> &fc,
    const tf::intersection_graph<Index, Int> &ig, const Forms &tagged_forms,
    const tf::exact::vertex_converter<Int, Real, Dims> &conv,
    const tf::buffer<VolT> &domain_volumes,
    const tf::buffer<char> &is_sheet_tag = {}) -> tf::buffer<Index> {
  using ag_t = tf::arrangement_graph<Index>;
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;
  using IntPt = tf::exact::pt3<Int>;
  using EVert = tf::exact::vertex<Index, Int>;

  const Index n_bundles = desc.n_bundles;
  const Index n_components = ag.n_components();
  const Index n_domains = desc.n_domains;
  const Index n_tags = ag.n_tags();
  const std::size_t words_per_domain = inc.words_per_domain;

  tf::buffer<Index> seeds;
  if (n_bundles == Index(0) || n_domains == Index(0) ||
      n_components == Index(0))
    return seeds;

  Index null_seed = Index(0);
  for (Index d = Index(1); d < n_domains; ++d)
    if (domain_volumes[d] < domain_volumes[null_seed])
      null_seed = d;

  tf::buffer<Index> outer_env;
  outer_env.allocate(static_cast<std::size_t>(n_bundles));
  for (Index b = Index(0); b < n_bundles; ++b)
    outer_env[b] = Index(-1);
  for (Index c = Index(0); c < n_components; ++c) {
    const Index b = desc.bundle_of_component[c];
    for (Index s = Index(0); s < Index(2); ++s) {
      const Index d = desc.domain_of_side[2 * c + s];
      if (outer_env[b] == Index(-1) ||
          domain_volumes[d] < domain_volumes[outer_env[b]])
        outer_env[b] = d;
    }
  }

  for (std::size_t w = 0; w < words_per_domain; ++w)
    inc.bits[static_cast<std::size_t>(null_seed) * words_per_domain + w] = 0u;

  auto record_seed = [&](Index d) {
    for (auto s : seeds)
      if (s == d)
        return;
    seeds.push_back(d);
  };
  record_seed(null_seed);

  if (n_bundles == Index(1))
    return seeds;

  // SoS id partition: originals [0, n_orig_total), createds
  // [n_orig_total, +n_created), per-bundle seeds above, far point last.
  // `conv.offsets` has size n_tags and holds *start* offsets, so the
  // total is the last form's start + its point count.
  const Index n_orig_total =
      Index(conv.offsets[static_cast<int>(n_tags - 1)]) +
      Index(tagged_forms[n_tags - 1].points().size());
  const Index n_created = Index(ig.points().size());
  const Index seed_id_base = n_orig_total + n_created;
  const Index far_id = seed_id_base + n_bundles;

  const Int int_max = std::numeric_limits<Int>::max();
  IntPt far_pt{int_max - Int(1), int_max - Int(2), int_max - Int(3)};

  auto get_original_vertex = [&](Index idx, Index tag) -> EVert {
    auto world = tf::transformed(tagged_forms[tag].points()[idx],
                                 tf::frame_of(tagged_forms[tag]));
    return {Index(idx) + Index(conv.offsets[static_cast<int>(tag)]),
            conv.convert(world)};
  };

  auto get_vertex = [&](const vertex_t &v, Index tag) -> EVert {
    if (v.source == source::created)
      return {n_orig_total + Index(v.id), ig.points()[v.id]};
    return get_original_vertex(Index(v.id), tag);
  };

  using bbox_t = tf::aabb<Int, 3>;
  auto bboxes = compute_bundle_aabbs(desc, ag, fc, ig, tagged_forms, conv);

  tf::buffer<IntPt> seed_pt;
  seed_pt.allocate(static_cast<std::size_t>(n_bundles));
  tf::buffer<char> seed_set;
  seed_set.allocate(static_cast<std::size_t>(n_bundles));
  for (Index b = Index(0); b < n_bundles; ++b)
    seed_set[b] = char(0);

  auto loops = fc.loops();
  auto descs = fc.descriptors();
  auto loop_labels = ag.loop_labels();
  const auto n_loops = loops.size();

  // Seed vertex per bundle: sequential walk that early-terminates as
  // soon as every bundle has any one vertex recorded. Worst case is
  // O(n_components) — way under the bbox work above.
  Index seeds_remaining = n_bundles;
  for (Index t = Index(0); t < n_tags && seeds_remaining > 0; ++t) {
    auto labels = ag.polygon_labels(t);
    auto faces = tagged_forms[t].faces();
    const auto n_faces = faces.size();
    for (std::size_t f = 0; f < n_faces && seeds_remaining > 0; ++f) {
      const Index c = labels[Index(f)];
      if (c == ag_t::none_label)
        continue;
      const Index b = desc.bundle_of_component[c];
      if (seed_set[b])
        continue;
      seed_pt[b] = get_original_vertex(Index(faces[Index(f)][0]), t).pt;
      seed_set[b] = char(1);
      --seeds_remaining;
    }
  }
  for (std::size_t l = 0; l < n_loops && seeds_remaining > 0; ++l) {
    const Index c = loop_labels[Index(l)];
    if (c == ag_t::none_label)
      continue;
    const Index tag = descs[Index(l)].tag;
    if (tag == Index(-1))
      continue;
    const Index b = desc.bundle_of_component[c];
    if (seed_set[b])
      continue;
    seed_pt[b] = get_vertex(loops[Index(l)][0], tag).pt;
    seed_set[b] = char(1);
    --seeds_remaining;
  }

  tf::buffer<bbox_t> form_bv;
  form_bv.allocate(static_cast<std::size_t>(n_tags));
  for (Index t = Index(0); t < n_tags; ++t) {
    auto local_bv = tagged_forms[t].tree().bv();
    auto world_bv = tf::transformed(local_bv, tf::frame_of(tagged_forms[t]));
    form_bv[t] =
        bbox_t{conv.convert(world_bv.min), conv.convert(world_bv.max)};
  }

  auto aabbs_overlap = [](const bbox_t &a, const bbox_t &b) -> bool {
    return a.min[0] <= b.max[0] && b.min[0] <= a.max[0] &&
           a.min[1] <= b.max[1] && b.min[1] <= a.max[1] &&
           a.min[2] <= b.max[2] && b.min[2] <= a.max[2];
  };

  auto sheet_tag = [&](Index t) -> bool {
    return t < Index(is_sheet_tag.size()) && is_sheet_tag[t];
  };

  // Skip whole forms whose tags are in bi's own bundle (binary search
  // on the ascending-sorted `desc.bundle_to_tags[bi]`). Sheets are
  // never AABB-skipped (a bundle has a side even without overlap) and
  // collect into their own per-sheet batches.
  tf::buffer<std::array<Index, 2>> candidates;
  tf::buffer<std::array<Index, 2>> sheet_pairs;
  for (Index bi = Index(0); bi < n_bundles; ++bi) {
    if (outer_env[bi] == null_seed)
      continue;
    if (!seed_set[bi])
      continue;
    auto own_tags = desc.bundle_to_tags[bi];
    for (Index t = Index(0); t < n_tags; ++t) {
      if (std::binary_search(own_tags.begin(), own_tags.end(), t))
        continue;
      if (sheet_tag(t))
        sheet_pairs.push_back({bi, t});
      else if (aabbs_overlap(bboxes[bi], form_bv[t]))
        candidates.push_back({bi, t});
    }
  }

  tf::buffer<char> parity;
  parity.allocate(static_cast<std::size_t>(n_bundles) *
                  static_cast<std::size_t>(n_tags));
  tf::parallel_fill(parity, char(0));

  // Sheet batches: sort the (bundle, sheet) pairs by sheet and group
  // by offsets; one winding pass per group classifies every bundle that
  // needs that sheet. Groups run in parallel and each pass is itself
  // parallel over the sheet's triangles. Writes are disjoint: group `t`
  // only touches parity column `t`.
  if (sheet_pairs.size() > 0) {
    tbb::parallel_sort(sheet_pairs.begin(), sheet_pairs.end(),
                       [](const auto &a, const auto &b) {
                         if (a[1] != b[1])
                           return a[1] < b[1];
                         return a[0] < b[0];
                       });
    tf::buffer<Index> sheet_offsets;
    sheet_offsets.reserve(sheet_pairs.size() + 1);
    tf::compute_offsets(
        sheet_pairs, std::back_inserter(sheet_offsets), Index(0),
        [](const auto &a, const auto &b) { return a[1] == b[1]; });
    auto groups =
        tf::make_offset_block_range(sheet_offsets, tf::make_range(sheet_pairs));
    tf::parallel_for_each(groups, [&](auto group) {
      const Index t = group[0][1];
      tf::buffer<IntPt> queries;
      queries.allocate(group.size());
      for (std::size_t j = 0; j < group.size(); ++j)
        queries[j] = seed_pt[group[j][0]];
      auto bits =
          winding_side(tagged_forms[t], queries,
                       [&](const auto &world) { return conv.convert(world); });
      for (std::size_t j = 0; j < bits.size(); ++j)
        parity[static_cast<std::size_t>(group[j][0]) *
                   static_cast<std::size_t>(n_tags) +
               static_cast<std::size_t>(t)] = bits[j];
    });
  }

  tf::parallel_for_each(candidates, [&](const auto &pair) {
    const Index bi = pair[0];
    const Index t = pair[1];
    const IntPt seed_pt_b = seed_pt[bi];
    char p = 0;

    EVert seed_v{seed_id_base + bi, seed_pt_b};
    EVert far_v{far_id, far_pt};

    auto labels = ag.polygon_labels(t);

    tf::search(
        tagged_forms[t],
        [&](const auto &world_float_bv) -> bool {
          auto lo = conv.convert(world_float_bv.min);
          auto hi = conv.convert(world_float_bv.max);
          return tf::exact::segment_hits_aabb(seed_pt_b, far_pt, lo, hi);
        },
        [&](const auto &tagged_poly) -> bool {
          const Index face_id = Index(tagged_poly.id());
          const Index c = labels[face_id];
          if (c != ag_t::none_label) {
            // Open patch (Mode-2 self-merged): d0 == d1, no transition.
            const Index d0 = desc.domain_of_side[2 * c + 0];
            const Index d1 = desc.domain_of_side[2 * c + 1];
            if (d0 == d1)
              return false;
          }
          auto face = tagged_forms[t].faces()[face_id];
          const auto n_fv = face.size();
          if (n_fv < 3)
            return false;
          auto v0 = get_original_vertex(Index(face[0]), t);
          for (std::size_t i = 1; i + 1 < n_fv; ++i) {
            auto va = get_original_vertex(Index(face[i]), t);
            auto vb = get_original_vertex(Index(face[i + 1]), t);
            std::array<EVert, 5> ts{v0, va, vb, seed_v, far_v};
            if (tf::exact::triangle_segment_intersect_point_sos(ts)) {
              p ^= char(1);
              break;
            }
          }
          return false;
        });

    parity[static_cast<std::size_t>(bi) * static_cast<std::size_t>(n_tags) +
           static_cast<std::size_t>(t)] = p;
  });

  // Bundles sharing an outer-env hit the same row; the first zeros it,
  // the rest OR in. Under the per-tag invariant their parity rows are
  // identical, so OR-ing matches the unique-writer semantics.
  for (Index bi = Index(0); bi < n_bundles; ++bi) {
    if (outer_env[bi] == null_seed)
      continue;
    if (!seed_set[bi])
      continue;
    const std::size_t row =
        static_cast<std::size_t>(outer_env[bi]) * words_per_domain;
    const bool first_writer = [&] {
      for (auto s : seeds)
        if (s == outer_env[bi])
          return false;
      return true;
    }();
    if (first_writer)
      for (std::size_t w = 0; w < words_per_domain; ++w)
        inc.bits[row + w] = 0u;
    for (Index t = Index(0); t < n_tags; ++t) {
      const char p =
          parity[static_cast<std::size_t>(bi) *
                     static_cast<std::size_t>(n_tags) +
                 static_cast<std::size_t>(t)];
      if (!p)
        continue;
      inc.bits[row + static_cast<std::size_t>(t) / 32u] |=
          std::uint32_t(1) << (static_cast<unsigned>(t) % 32u);
    }
    record_seed(outer_env[bi]);
  }

  return seeds;
}

} // namespace tf::csg::graph
