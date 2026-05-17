/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/int64.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/triangle_segment_intersection.hpp"
#include "../../exact/vertex.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <limits>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Compute domain-id merge pairs from bundle nesting.
///
/// For each 3D-connected bundle, identifies its outer-environment domain
/// (the incident domain with the most-negative volume) and ray-casts a
/// seed vertex through the global mesh against every face NOT in
/// `b_inner`. Each face-segment intersection is registered as a hit in
/// **both** of the face's adjacent domains (a face touches exactly two
/// domains). Domains with odd hit count are exactly those that contain
/// the seed (the seed's own domain plus every enclosing one); among
/// those, the one whose closest hit lies nearest the seed is the
/// seed's actual physical domain.
///
/// Open patches (Mode 2 boundary self-merged) have both sides bounding
/// the same domain and contribute no transition --- skipped.
///
/// Two kinds of merges are produced:
///   - **Nesting merges**: `(b_inner.outer_env_domain,
///     seed_domain)` for each `b_inner` enclosed by some other bundle.
///     `seed_domain` is the closest-hit's incident domain with odd
///     parity --- the domain the seed physically sits in.
///   - **Universe merges**: bundles whose outer-env was *not* enclosed
///     by anyone are "roots"; all their outer-env domains merge into
///     one anchor (the true universe).
template <typename Int = tf::exact::int64, typename Polygons,
          typename FragLabels, typename BundleLabels, typename Index,
          typename VolType, typename GetPoint>
auto make_nesting_merges(const Polygons &polygons,
                         const FragLabels &fragment_labels,
                         const BundleLabels &bundle_labels,
                         const tf::buffer<Index> &domain_of_side,
                         const tf::buffer<VolType> &domain_volumes,
                         const GetPoint &get_point,
                         tf::buffer<std::array<Index, 2>> &merges,
                         Index removed_domain = Index(-1)) -> Index {
  using IntPt = tf::exact::pt3<Int>;
  using Vert = tf::exact::vertex<Index, Int>;
  using PairT = std::array<Index, 2>;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  struct bbox_t {
    IntPt mn;
    IntPt mx;
  };

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

  Index n_bundles = bundle_labels.n_components;
  Index n_components = fragment_labels.n_components;
  Index n_faces = Index(polygons.faces().size());
  Index n_points = Index(polygons.points().size());

  if (n_bundles == Index(0))
    return Index(-1);

  tf::buffer<Index> outer_env;
  outer_env.allocate(n_bundles);
  tf::parallel_fill(outer_env, Index(-1));

  tf::buffer<VolType> best_vol;
  best_vol.allocate(n_bundles);

  for (Index c = Index(0); c < n_components; ++c) {
    Index b = bundle_labels.labels[c];
    for (Index s = Index(0); s < Index(2); ++s) {
      Index d = domain_of_side[2 * c + s];
      const auto &v = domain_volumes[d];
      if (outer_env[b] == Index(-1) || v < best_vol[b]) {
        outer_env[b] = d;
        best_vol[b] = v;
      }
    }
  }

  if (n_bundles == Index(1))
    return (outer_env[0] == removed_domain) ? Index(-1) : outer_env[0];

  // A face-side landing on a shell domain contributes no hit: only
  // bounded interior domains are tracked. Shells = per-bundle outer-env
  // + the Mode-1 removed domain.
  Index n_domains = Index(domain_volumes.size());
  tf::buffer<char> is_shell;
  is_shell.allocate(n_domains);
  tf::parallel_fill(is_shell, char(0));
  for (Index b = Index(0); b < n_bundles; ++b)
    if (outer_env[b] >= Index(0))
      is_shell[outer_env[b]] = char(1);
  if (removed_domain >= Index(0))
    is_shell[removed_domain] = char(1);

  tf::buffer<Index> seed_v;
  seed_v.allocate(n_bundles);
  tf::parallel_fill(seed_v, Index(-1));
  for (Index f = Index(0); f < n_faces; ++f) {
    Index c = fragment_labels.labels[f];
    Index b = bundle_labels.labels[c];
    if (seed_v[b] == Index(-1))
      seed_v[b] = Index(polygons.faces()[f][0]);
  }

  Int int_max = std::numeric_limits<Int>::max();
  Int int_min = std::numeric_limits<Int>::min();
  bbox_t init_bbox{IntPt{int_max, int_max, int_max},
                   IntPt{int_min, int_min, int_min}};

  tf::buffer<bbox_t> bboxes;
  bboxes.allocate(n_bundles);
  for (Index b = Index(0); b < n_bundles; ++b)
    bboxes[b] = init_bbox;

  // Per-thread bbox accumulator inline up to 8 bundles (no heap alloc
  // in the typical case).
  tf::small_vector<bbox_t, 8> bbox_proto(n_bundles, init_bbox);

  tf::blocked_reduce_sequenced_aggregate(
      tf::enumerate(polygons.faces()), bboxes, bbox_proto,
      [&](auto &&block, tf::small_vector<bbox_t, 8> &local) {
        for (const auto &pair : block) {
          const auto &[f, face] = pair;
          Index c = fragment_labels.labels[f];
          Index b = bundle_labels.labels[c];
          for (auto v_idx : face) {
            auto p = get_point(Index(v_idx));
            for (int k = 0; k < 3; ++k) {
              local[b].mn[k] = std::min(local[b].mn[k], p[k]);
              local[b].mx[k] = std::max(local[b].mx[k], p[k]);
            }
          }
        }
      },
      [](const tf::small_vector<bbox_t, 8> &local, tf::buffer<bbox_t> &result) {
        for (std::size_t b = 0; b < result.size(); ++b) {
          for (int k = 0; k < 3; ++k) {
            result[b].mn[k] = std::min(result[b].mn[k], local[b].mn[k]);
            result[b].mx[k] = std::max(result[b].mx[k], local[b].mx[k]);
          }
        }
      });

  // Far point past the 99%-of-int_max pt_converter range, guaranteed
  // outside the global bbox. Unique SoS vertex id = n_points.
  IntPt p_far_pt{int_max - Int(1), int_max - Int(2), int_max - Int(3)};
  Vert p_far_v{n_points, p_far_pt};

  tf::buffer<PairT> candidates;
  for (Index bi = Index(0); bi < n_bundles; ++bi) {
    if (outer_env[bi] == removed_domain)
      continue;
    for (Index bo = Index(0); bo < n_bundles; ++bo) {
      if (bi == bo)
        continue;
      if (outer_env[bo] == removed_domain)
        continue;
      const auto &Bi = bboxes[bi];
      const auto &Bo = bboxes[bo];
      bool inside = Bi.mn[0] >= Bo.mn[0] && Bi.mx[0] <= Bo.mx[0] &&
                    Bi.mn[1] >= Bo.mn[1] && Bi.mx[1] <= Bo.mx[1] &&
                    Bi.mn[2] >= Bo.mn[2] && Bi.mx[2] <= Bo.mx[2];
      if (inside)
        candidates.push_back({bi, bo});
    }
  }

  tf::buffer<hit_t> hits;
  tf::generic_generate(
      tf::make_sequence_range(n_faces), hits,
      [&](Index f, tf::buffer<hit_t> &out) {
        Index c = fragment_labels.labels[f];
        Index b_face = bundle_labels.labels[c];
        Index d0 = domain_of_side[2 * c + 0];
        Index d1 = domain_of_side[2 * c + 1];
        if (d0 == d1)
          return;
        if (is_shell[d0] && is_shell[d1])
          return;

        auto face = polygons.faces()[f];
        auto n_fv = Index(face.size());
        if (n_fv < Index(3))
          return;

        Vert v0{Index(face[0]), get_point(Index(face[0]))};

        for (auto pair : candidates) {
          Index bi = pair[0];
          Index bo = pair[1];
          if (bo != b_face)
            continue;
          Index seed_id = seed_v[bi];
          if (seed_id == Index(-1))
            continue;
          IntPt seed_pt = get_point(seed_id);
          Vert seed_vert{seed_id, seed_pt};

          for (Index t = Index(1); t + Index(1) < n_fv; ++t) {
            Vert va{Index(face[t]), get_point(Index(face[t]))};
            Vert vb{Index(face[t + 1]), get_point(Index(face[t + 1]))};
            std::array<Vert, 5> tri_seg{v0, va, vb, seed_vert, p_far_v};
            if (auto hit_opt = tf::exact::triangle_segment_intersect_point_sos(
                    tri_seg)) {
              auto hit = *hit_opt;
              T1 dx = T1(hit[0]) - T1(seed_pt[0]);
              T1 dy = T1(hit[1]) - T1(seed_pt[1]);
              T1 dz = T1(hit[2]) - T1(seed_pt[2]);
              T2 dist_sq = T2(dx) * T2(dx) + T2(dy) * T2(dy) + T2(dz) * T2(dz);
              if (!is_shell[d0])
                out.push_back({bi, d0, dist_sq});
              if (!is_shell[d1])
                out.push_back({bi, d1, dist_sq});
              break;
            }
          }
        }
      });

  tbb::parallel_sort(hits.begin(), hits.end());

  tf::buffer<Index> chosen_target;
  chosen_target.allocate(n_bundles);
  tf::parallel_fill(chosen_target, Index(-1));

  // best_dist_sq is only read when chosen_target[bi] != -1.
  tf::buffer<T2> best_dist_sq;
  best_dist_sq.allocate(n_bundles);

  for (auto it = hits.begin(); it != hits.end();) {
    Index bi = it->b_inner;
    Index d = it->domain;
    auto group_end = it;
    std::size_t cnt = 0;
    while (group_end != hits.end() && group_end->b_inner == bi &&
           group_end->domain == d) {
      ++cnt;
      ++group_end;
    }
    T2 closest = it->dist_sq;
    if ((cnt & 1u) == 1u) {
      if (chosen_target[bi] == Index(-1) || closest < best_dist_sq[bi]) {
        best_dist_sq[bi] = closest;
        chosen_target[bi] = d;
      }
    }
    it = group_end;
  }

  for (Index bi = Index(0); bi < n_bundles; ++bi) {
    Index d_out = chosen_target[bi];
    if (d_out == Index(-1))
      continue;
    Index d_in = outer_env[bi];
    if (d_in == Index(-1) || d_in == d_out)
      continue;
    merges.push_back({std::min(d_in, d_out), std::max(d_in, d_out)});
  }

  // Root bundles (no parent) merge their outer-envs into one anchor.
  Index root_anchor = Index(-1);
  for (Index bi = Index(0); bi < n_bundles; ++bi) {
    if (chosen_target[bi] != Index(-1))
      continue;
    Index d = outer_env[bi];
    if (d == Index(-1) || d == removed_domain)
      continue;
    if (root_anchor == Index(-1)) {
      root_anchor = d;
    } else if (d != root_anchor) {
      merges.push_back(
          {std::min(d, root_anchor), std::max(d, root_anchor)});
    }
  }

  return root_anchor;
}

} // namespace tf::topology::domains
