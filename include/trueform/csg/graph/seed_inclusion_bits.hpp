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
#include "../../core/aabb.hpp"
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/inflated_aabb.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/segment_hits_aabb.hpp"
#include "../../exact/triangle_segment_intersection.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../spatial/aabb_from.hpp"
#include "../../spatial/search.hpp"
#include "./arrangement_descriptor.hpp"
#include "./compute_bundle_aabbs.hpp"
#include "./domain_inclusions.hpp"
#include "./structural_membership.hpp"
#include "./triangle_component_labels.hpp"
#include "./winding_side.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Seed per-bundle outer-env inclusion bits AND per-bundle nesting
///        merges from a single per-bundle raycast (one walk, two reductions).
///
/// Each disconnected bundle's seed is segment-cast to a far point against
/// per-form AABB trees (`tf::search` with exact `segment_hits_aabb` at
/// every BV node, exact `triangle_segment_intersect_point_sos` at leaves).
/// That cast is reduced two ways:
///   1. **bits** — per-form parity into `inc.bits[outer_env(bi)]`. Each
///      bundle's outer-shell is the most-negative-volume incident domain;
///      the globally most-negative is `null_seed`, anchored at zero bits.
///   2. **nesting** (`out_nesting_merges`) — from each hit's two adjacent
///      domains + squared seed-distance, the closest odd-parity domain is
///      the seed's physical container; `(outer_env[bi], that_domain)` is
///      emitted as a `domain_of_side` merge that repairs the false split
///      the arrangement leaves between two contact-free nested shells (the
///      implicit-arrangement analogue of
///      @ref tf::topology::domains::make_nesting_merges). No extra cast.
///
/// Sheet forms (`is_sheet_tag`) are side-classified via the generalized
/// winding number (@ref tf::csg::graph::winding_side) instead of raycast;
/// they never enclose, so they contribute no nesting merge. Single-bundle
/// inputs return before any nesting work.
template <typename Index, typename Int, typename Arrangement,
          typename ApplyToForm, typename Real, std::size_t Dims, typename VolT,
          typename GetMeshPoint>
auto seed_inclusion_bits(
    tf::csg::graph::domain_inclusions &inc,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const ApplyToForm &apply_to_form,
    const tf::exact::vertex_converter<Int, Real, Dims> &conv,
    const GetMeshPoint &get_mesh_point, const tf::buffer<VolT> &domain_volumes,
    tf::buffer<std::array<Index, 2>> &out_nesting_merges,
    const tf::buffer<char> &is_sheet_tag = {}) -> tf::buffer<Index> {
  using labels_t = tf::csg::graph::triangle_component_labels<Index>;
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;
  using IntPt = tf::exact::pt3<Int>;
  using EVert = tf::exact::vertex<Index, Int>;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  struct hit_t {
    Index b_inner;
    Index domain;
    T2 dist_sq;
    auto operator<(const hit_t &o) const -> bool {
      if (b_inner != o.b_inner)
        return b_inner < o.b_inner;
      if (domain != o.domain)
        return domain < o.domain;
      return dist_sq < o.dist_sq;
    }
  };

  out_nesting_merges.clear();

  const Index n_bundles = desc.n_bundles;
  const Index n_components = labels.n_components();
  const Index n_domains = desc.n_domains;
  const Index n_tags = arrangement.n_tags();
  const std::size_t words_per_domain = inc.words_per_domain;

  tf::buffer<Index> seeds;
  if (n_bundles == Index(0) || n_domains == Index(0) ||
      n_components == Index(0))
    return seeds;

  Index null_seed =
      Index(tf::csg::graph::find_universe_domain(domain_volumes));

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

  tf::buffer<char> is_shell;
  is_shell.allocate(static_cast<std::size_t>(n_domains));
  tf::parallel_fill(is_shell, char(0));
  for (Index b = Index(0); b < n_bundles; ++b)
    if (outer_env[b] >= Index(0))
      is_shell[outer_env[b]] = char(1);

  // SoS id partition: originals [0, n_orig_total), createds
  // [n_orig_total, +n_created), per-bundle seeds above, far point last.
  // The stream's original ids ARE the flat ids, so the flat space is
  // the original partition verbatim.
  auto voffs = tf::make_range(arrangement.vertex_offsets());
  const auto &created = arrangement.created_points();
  const Index n_orig_total = voffs[voffs.size() - 1];
  const Index n_created = Index(created.size());
  const Index seed_id_base = n_orig_total + n_created;
  const Index far_id = seed_id_base + n_bundles;

  const Int int_max = std::numeric_limits<Int>::max();
  IntPt far_pt{int_max - Int(1), int_max - Int(2), int_max - Int(3)};

  const auto &ga = arrangement.global();

  auto get_original_vertex = [&](Index idx, Index tag) -> EVert {
    return {voffs[std::size_t(tag)] + idx, get_mesh_point(int(tag), idx)};
  };

  // `own_tag` = the carrier's tag when its plane has a single member
  // (its originals are that form's), -1 when the pooled CDT may have
  // handed the triangle another form's corner.
  auto get_vertex = [&](const vertex_t &v, Index own_tag) -> EVert {
    if (v.source == source::created)
      return {n_orig_total + Index(v.id), created[std::size_t(v.id)]};
    const auto tg = own_tag != Index(-1) ? own_tag : arrangement.tag_of_flat(v.id);
    return {Index(v.id),
            get_mesh_point(int(tg), Index(v.id - voffs[std::size_t(tg)]))};
  };

  using bbox_t = tf::aabb<Int, 3>;
  auto bboxes = compute_bundle_aabbs<Index, Int>(
      desc, arrangement, labels, apply_to_form, get_mesh_point);

  tf::buffer<IntPt> seed_pt;
  seed_pt.allocate(static_cast<std::size_t>(n_bundles));
  tf::buffer<char> seed_set;
  seed_set.allocate(static_cast<std::size_t>(n_bundles));
  for (Index b = Index(0); b < n_bundles; ++b)
    seed_set[b] = char(0);

  auto descs = ga.exposed_descriptors();
  auto tris = ga.exposed_tris();
  auto tri_labels = labels.triangle_labels();
  auto tri_tags = arrangement.triangle_tags();
  auto slots = arrangement.triangle_slots();
  auto pooled = arrangement.pooled_slots();
  auto own_tag_of = [&](Index e) {
    return pooled[slots[e]] ? Index(-1) : tri_tags[e];
  };

  // Seed vertex per bundle: sequential walk that early-terminates as
  // soon as every bundle has any one vertex recorded.
  Index seeds_remaining = n_bundles;
  for (Index t = Index(0); t < n_tags && seeds_remaining > 0; ++t)
    apply_to_form(t, [&](const auto &form) {
      auto polygon_labels = labels.polygon_labels(t);
      auto faces = form.faces();
      const auto n_faces = faces.size();
      for (std::size_t f = 0; f < n_faces && seeds_remaining > 0; ++f) {
        const Index c = polygon_labels[Index(f)];
        if (c == labels_t::none_label)
          continue;
        const Index b = desc.bundle_of_component[c];
        if (seed_set[b])
          continue;
        seed_pt[b] = get_original_vertex(Index(faces[Index(f)][0]), t).pt;
        seed_set[b] = char(1);
        --seeds_remaining;
      }
    });
  for (Index e = Index(0); e < Index(tris.size()) && seeds_remaining > 0; ++e) {
    const Index c = tri_labels[e];
    if (c == labels_t::none_label)
      continue;
    const Index b = desc.bundle_of_component[c];
    if (seed_set[b])
      continue;
    seed_pt[b] = get_vertex(tris[e][0], own_tag_of(e)).pt;
    seed_set[b] = char(1);
    --seeds_remaining;
  }

  // THE CAST'S BOXES ARE THE INPUT'S; ITS HITS ARE THE PLACED MESH'S. A
  // pruned hit flips a parity bit and misclassifies a whole domain, so
  // every box the cast prunes against grows by the door's motion bound.
  const Int reach = get_mesh_point.motion_bound;
  tf::buffer<bbox_t> form_bv;
  form_bv.allocate(static_cast<std::size_t>(n_tags));
  for (Index t = Index(0); t < n_tags; ++t)
    apply_to_form(t, [&](const auto &form) {
      auto local_bv = tf::aabb_from(form.tree());
      auto world_bv = tf::transformed(local_bv, tf::frame_of(form));
      form_bv[t] = tf::inflated_aabb(
          bbox_t{conv.convert(world_bv.min), conv.convert(world_bv.max)},
          reach);
    });

  auto aabbs_overlap = [](const bbox_t &a, const bbox_t &b) -> bool {
    return a.min[0] <= b.max[0] && b.min[0] <= a.max[0] &&
           a.min[1] <= b.max[1] && b.min[1] <= a.max[1] &&
           a.min[2] <= b.max[2] && b.min[2] <= a.max[2];
  };

  auto sheet_tag = [&](Index t) -> bool {
    return t < Index(is_sheet_tag.size()) && is_sheet_tag[t];
  };

  tf::buffer<std::array<Index, 2>> candidates;
  tf::buffer<std::array<Index, 2>> sheet_pairs;
  bool any_own_candidate = false;
  for (Index bi = Index(0); bi < n_bundles; ++bi) {
    if (outer_env[bi] == null_seed)
      continue;
    if (!seed_set[bi])
      continue;
    auto own_tags = desc.bundle_to_tags[bi];
    for (Index t = Index(0); t < n_tags; ++t) {
      const bool own = std::binary_search(own_tags.begin(), own_tags.end(), t);
      if (sheet_tag(t)) {
        if (!own)
          sheet_pairs.push_back({bi, t});
      } else if (aabbs_overlap(bboxes[bi], form_bv[t])) {
        candidates.push_back({bi, t});
        any_own_candidate = any_own_candidate || own;
      }
    }
  }

  // A casting bundle must not parity-count its own surface. Uncut own
  // faces are skipped by component label in the cast callback; a CUT
  // face carries `none_label` there, so its bundle comes from its
  // triangles — every piece of a face lies on one original surface,
  // hence one bundle.
  // (tag, face) -> exposed slot: exposure is tag-major, object-dense,
  // so the prefix of per-tag face counts is the slot base
  tf::buffer<Index> face_slot_offsets;
  face_slot_offsets.push_back(Index(0));
  for (Index t = Index(0); t < n_tags; ++t)
    apply_to_form(t, [&](const auto &form) {
      face_slot_offsets.push_back(face_slot_offsets[std::size_t(t)] +
                                  Index(form.faces().size()));
    });
  tf::buffer<Index> cut_face_bundle;
  if (any_own_candidate) {
    cut_face_bundle.allocate(
        static_cast<std::size_t>(face_slot_offsets[std::size_t(n_tags)]));
    tf::parallel_fill(cut_face_bundle, Index(-1));
    for (Index e = Index(0); e < Index(tris.size()); ++e) {
      const Index c = tri_labels[e];
      if (c == labels_t::none_label)
        continue;
      const auto &d = descs[slots[e]];
      cut_face_bundle[static_cast<std::size_t>(
          face_slot_offsets[std::size_t(Index(d.tag))] + d.object)] =
          desc.bundle_of_component[c];
    }
  }
  auto refine_promoted = arrangement.refine_promoted();

  tf::buffer<char> parity;
  parity.allocate(static_cast<std::size_t>(n_bundles) *
                  static_cast<std::size_t>(n_tags));
  tf::parallel_fill(parity, char(0));

  // Sheet batches: one winding pass per sheet classifies every bundle
  // that needs it; group `t` only touches parity column `t`.
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
      apply_to_form(t, [&, t](const auto &form) {
        auto bits = tf::csg::graph::winding_side(
            form, queries,
            [&, t](Index id) { return get_mesh_point(int(t), id); });
        for (std::size_t j = 0; j < bits.size(); ++j)
          parity[static_cast<std::size_t>(group[j][0]) *
                     static_cast<std::size_t>(n_tags) +
                 static_cast<std::size_t>(t)] = bits[j];
      });
    });
  }

  // One cast, two reductions: per-form parity (the bits) AND nesting
  // hits (each hit's two adjacent bounded domains + squared
  // seed-distance).
  tf::buffer<hit_t> nesting_hits;
  tf::generic_generate(
      tf::make_range(candidates), nesting_hits,
      [&](const auto &pair, tf::buffer<hit_t> &out) {
        const Index bi = pair[0];
        const Index t = pair[1];
        const IntPt seed_pt_b = seed_pt[bi];
        char p = 0;

        EVert seed_v{seed_id_base + bi, seed_pt_b};
        EVert far_v{far_id, far_pt};

        auto polygon_labels = labels.polygon_labels(t);

        apply_to_form(t, [&, t](const auto &form) {
          const Index id_offset = voffs[std::size_t(t)];
          auto vertex_of = [&, t](Index idx) -> EVert {
            return {idx + id_offset, get_mesh_point(int(t), idx)};
          };
          tf::search(
              form,
              [&](const auto &world_float_bv) -> bool {
                const auto box = tf::inflated_aabb(
                    tf::make_aabb(conv.convert(world_float_bv.min),
                                  conv.convert(world_float_bv.max)),
                    reach);
                return tf::exact::segment_hits_aabb(seed_pt_b, far_pt, box.min,
                                                    box.max);
              },
              [&](const auto &tagged_poly) -> bool {
                const Index face_id = Index(tagged_poly.id());
                const Index c = polygon_labels[face_id];
                Index d0 = Index(-1), d1 = Index(-1);
                bool emit = false;
                if (c != labels_t::none_label) {
                  if (desc.bundle_of_component[c] == bi)
                    return false;
                  // Open patch (Mode-2 self-merged): d0 == d1, no transition.
                  d0 = desc.domain_of_side[2 * c + 0];
                  d1 = desc.domain_of_side[2 * c + 1];
                  if (d0 == d1)
                    return false;
                  emit = true;
                } else {
                  if (cut_face_bundle.size() > 0 &&
                      cut_face_bundle[static_cast<std::size_t>(
                          face_slot_offsets[std::size_t(t)] + face_id)] ==
                          bi)
                    return false;
                  // A refine-promoted face is uncut in truth — one
                  // plane, one stream component. Its domain transition
                  // enters the census exactly as the uncut original's
                  // did, or hits through it vanish and refined nesting
                  // parity diverges from stock.
                  if (refine_promoted.size() != 0 &&
                      std::binary_search(
                          refine_promoted.begin(), refine_promoted.end(),
                          std::array<Index, 2>{t, face_id})) {
                    const auto rg = arrangement.slot_range(
                        face_slot_offsets[std::size_t(t)] + face_id);
                    if (rg[0] != rg[1]) {
                      const Index c2 = tri_labels[rg[0]];
                      if (c2 != labels_t::none_label) {
                        if (desc.bundle_of_component[c2] == bi)
                          return false;
                        d0 = desc.domain_of_side[2 * c2 + 0];
                        d1 = desc.domain_of_side[2 * c2 + 1];
                        if (d0 == d1)
                          return false;
                        emit = true;
                      }
                    }
                  }
                }
                auto face = form.faces()[face_id];
                const auto n_fv = face.size();
                if (n_fv < 3)
                  return false;
                auto v0 = vertex_of(Index(face[0]));
                for (std::size_t i = 1; i + 1 < n_fv; ++i) {
                  auto va = vertex_of(Index(face[i]));
                  auto vb = vertex_of(Index(face[i + 1]));
                  std::array<EVert, 5> ts{v0, va, vb, seed_v, far_v};
                  if (auto hit_opt =
                          tf::exact::triangle_segment_intersect_point_sos(ts)) {
                    p ^= char(1);
                    if (emit) {
                      auto hit = *hit_opt;
                      const T1 dx = T1(hit[0]) - T1(seed_pt_b[0]);
                      const T1 dy = T1(hit[1]) - T1(seed_pt_b[1]);
                      const T1 dz = T1(hit[2]) - T1(seed_pt_b[2]);
                      const T2 dist_sq =
                          T2(dx) * T2(dx) + T2(dy) * T2(dy) + T2(dz) * T2(dz);
                      if (!is_shell[d0])
                        out.push_back({bi, d0, dist_sq});
                      if (!is_shell[d1])
                        out.push_back({bi, d1, dist_sq});
                    }
                    break;
                  }
                }
                return false;
              });
        });

        parity[static_cast<std::size_t>(bi) * static_cast<std::size_t>(n_tags) +
               static_cast<std::size_t>(t)] = p;
      });

  // Bundles sharing an outer-env hit the same row; the first zeros it,
  // the rest OR in.
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
      inc.set(static_cast<std::size_t>(outer_env[bi]),
              static_cast<std::size_t>(t));
    }
    record_seed(outer_env[bi]);
  }

  // ---- Nesting reduction over the hits the cast already produced. ----
  // Per (bundle, domain): odd hit-parity ⇒ the domain encloses the seed;
  // among those the closest is the seed's physical domain.
  tbb::parallel_sort(nesting_hits.begin(), nesting_hits.end());

  tf::buffer<Index> chosen_target;
  chosen_target.allocate(static_cast<std::size_t>(n_bundles));
  tf::parallel_fill(chosen_target, Index(-1));
  tf::buffer<T2> best_dist_sq; // read only where chosen_target[bi] != -1
  best_dist_sq.allocate(static_cast<std::size_t>(n_bundles));

  for (auto it = nesting_hits.begin(); it != nesting_hits.end();) {
    const Index bi = it->b_inner;
    const Index d = it->domain;
    auto group_end = it;
    std::size_t cnt = 0;
    while (group_end != nesting_hits.end() && group_end->b_inner == bi &&
           group_end->domain == d) {
      ++cnt;
      ++group_end;
    }
    const T2 closest = it->dist_sq; // groups are sorted by dist_sq
    if ((cnt & 1u) == 1u) {
      if (chosen_target[bi] == Index(-1) || closest < best_dist_sq[bi]) {
        best_dist_sq[bi] = closest;
        chosen_target[bi] = d;
      }
    }
    it = group_end;
  }

  for (Index bi = Index(0); bi < n_bundles; ++bi) {
    const Index d_in = outer_env[bi];
    if (d_in == Index(-1))
      continue;
    // No enclosing domain means the bundle's envelope reaches infinity:
    // it is the global outside, whatever bits side-anchoring gave it.
    const Index d_out =
        chosen_target[bi] == Index(-1) ? null_seed : chosen_target[bi];
    if (d_in == d_out)
      continue;
    out_nesting_merges.push_back(
        {std::min(d_in, d_out), std::max(d_in, d_out)});
  }

  return seeds;
}

} // namespace tf::csg::graph
