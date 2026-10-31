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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/point.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/triangles_cross.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// THE ROUNDING CROSSINGS the near table witnesses: every two triangles of
/// the arrangement `ag` on different carriers, one in the star of each
/// member of a near pair, that cross (@ref tf::exact::triangles_cross).
///
/// `pairs` is the near table as its members speak now (@ref
/// make_live_near_identities). A star is the live exposed triangles naming
/// the member as a corner, and an original's star also holds the uncut
/// triangles its face membership names, each as `n_exposed + slot`; the
/// exact test reads triangles, so an uncut face of more corners witnesses
/// nothing. Two triangles of one carrier come out of one triangulation, so
/// they never cross; two sharing an edge cannot cross strictly; two uncut
/// faces are the input's own, which the intersection tier states. `crossed`
/// receives `{T, U, a, b}` per crossing, `T < U` ascending and once, `(a, b)`
/// the first near pair that witnessed it; an empty near table asks nothing.
template <typename ArrangementGraph>
auto find_census_crossings(
    const ArrangementGraph &ag,
    const tf::buffer<std::array<typename ArrangementGraph::index_type, 2>>
        &pairs,
    tf::buffer<std::array<typename ArrangementGraph::index_type, 4>> &crossed)
    -> void {
  using Index = typename ArrangementGraph::index_type;
  using Int = typename ArrangementGraph::resolved_int_type;
  using pt3 = tf::point<Int, 3>;
  using record_t = std::array<Index, 4>;
  using vsource = tf::intersect::graph::vertex_source;
  crossed.clear();
  if (pairs.size() == 0)
    return;

  const auto &arr = ag.arrangement();
  const auto n_flat = arr.n_flat_points();
  const auto &created = ag.created_points();

  // the members, ascending and once: a member's star is its block
  tf::buffer<Index> members;
  members.allocate(pairs.size() * 2);
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    members[2 * k] = pairs[k][0];
    members[2 * k + 1] = pairs[k][1];
  }
  tbb::parallel_sort(members.begin(), members.end());
  members.erase_till_end(std::unique(members.begin(), members.end()));
  tf::buffer<char> marked;
  marked.allocate(std::size_t(n_flat) + created.size());
  tf::parallel_fill(marked, char(0));
  for (const auto id : members)
    marked[std::size_t(id)] = char(1);

  const auto tris = ag.global().exposed_tris();
  const auto descriptors = ag.global().exposed_descriptors();
  const auto slot_base = ag.global().face_slot_base();
  const auto &vertex_offsets = ag.vertex_offsets();
  const auto n_exposed = Index(tris.size());
  const auto dead = ag.dead();
  const auto row_of = ag.row_of();
  const auto apply_form = ag.apply_to_form();
  const auto reader = ag.lattice().reader(apply_form);
  const auto flat_of_corner = [n_flat](const auto &v) -> Index {
    return v.source == vsource::original ? v.id : n_flat + v.id;
  };

  // {member, triangle}: the stream's corners, then the uncut faces an
  // original's membership names
  tf::buffer<std::array<Index, 2>> stars;
  tf::generic_generate(
      tf::make_sequence_range(n_exposed), stars,
      [&](Index e, tf::buffer<std::array<Index, 2>> &out) {
        if (dead[e])
          return;
        for (std::size_t c = 0; c < 3; ++c) {
          const auto flat = flat_of_corner(tris[std::size_t(e)][c]);
          if (marked[std::size_t(flat)])
            out.push_back({flat, e});
        }
      },
      tf::checked);
  tf::generic_generate(
      members, stars,
      [&](Index flat, tf::buffer<std::array<Index, 2>> &out) {
        if (flat >= n_flat)
          return;
        const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
        const auto local = flat - vertex_offsets[std::size_t(tag)];
        apply_form(Index(tag), [&](const auto &form) {
          if constexpr (tf::has_face_membership_policy<decltype(form)>)
            for (const auto face : form.face_membership()[local]) {
              const auto slot = slot_base[std::size_t(tag)] + Index(face);
              if (descriptors[std::size_t(slot)].plane == Index(-1) &&
                  form.faces()[face].size() == 3)
                out.push_back({flat, n_exposed + slot});
            }
        });
      },
      tf::checked);
  tbb::parallel_sort(stars.begin(), stars.end());
  tf::buffer<Index> star_offsets;
  star_offsets.allocate(members.size() + 1);
  tf::parallel_transform(
      tf::make_sequence_range(members.size() + 1), star_offsets,
      [&](std::size_t k) {
        if (k == members.size())
          return Index(stars.size());
        return Index(
            std::lower_bound(stars.begin(), stars.end(),
                             std::array<Index, 2>{members[k], Index(-1)}) -
            stars.begin());
      },
      tf::checked);
  const auto star_of = [&](Index id) {
    const auto k = std::size_t(
        std::lower_bound(members.begin(), members.end(), id) - members.begin());
    return std::array<std::size_t, 2>{std::size_t(star_offsets[k]),
                                      std::size_t(star_offsets[k + 1])};
  };
  const auto carrier = [&](Index t) -> Index {
    return t < n_exposed ? arr.plane_of_triangle(row_of[t]) : Index(-1);
  };

  tf::buffer<record_t> candidates;
  tf::generic_generate(
      pairs, candidates,
      [&](const std::array<Index, 2> &pair, tf::buffer<record_t> &out) {
        const auto sa = star_of(pair[0]);
        const auto sb = star_of(pair[1]);
        for (auto i = sa[0]; i < sa[1]; ++i)
          for (auto j = sb[0]; j < sb[1]; ++j) {
            const auto t = stars[i][1];
            const auto u = stars[j][1];
            if (t == u || (t >= n_exposed && u >= n_exposed) ||
                carrier(t) == carrier(u))
              continue;
            out.push_back(t < u ? record_t{t, u, pair[0], pair[1]}
                                : record_t{u, t, pair[0], pair[1]});
          }
      },
      tf::checked);
  tbb::parallel_sort(candidates.begin(), candidates.end());
  candidates.erase_till_end(
      std::unique(candidates.begin(), candidates.end(),
                  [](const record_t &x, const record_t &y) {
                    return x[0] == y[0] && x[1] == y[1];
                  }));

  struct corners_t {
    std::array<Index, 3> flats;
    std::array<pt3, 3> points;
  };
  const auto corners_of = [&](Index t) {
    corners_t out;
    if (t < n_exposed) {
      for (std::size_t c = 0; c < 3; ++c) {
        const auto &v = tris[std::size_t(t)][c];
        out.flats[c] = flat_of_corner(v);
        out.points[c] = v.source == vsource::original
                            ? pt3(reader.point_of_flat(v.id))
                            : pt3(created[std::size_t(v.id)]);
      }
      return out;
    }
    const auto &descriptor = descriptors[std::size_t(t - n_exposed)];
    apply_form(Index(descriptor.tag), [&](const auto &form) {
      const auto face = form.faces()[descriptor.object];
      for (std::size_t c = 0; c < 3; ++c) {
        out.flats[c] =
            vertex_offsets[std::size_t(descriptor.tag)] + Index(face[c]);
        out.points[c] = pt3(reader(int(descriptor.tag), Index(face[c])));
      }
    });
    return out;
  };
  const auto boxes_meet = [](const std::array<pt3, 3> &p,
                             const std::array<pt3, 3> &q) {
    for (std::size_t k = 0; k < 3; ++k) {
      const auto p_lo = std::min({p[0][k], p[1][k], p[2][k]});
      const auto p_hi = std::max({p[0][k], p[1][k], p[2][k]});
      const auto q_lo = std::min({q[0][k], q[1][k], q[2][k]});
      const auto q_hi = std::max({q[0][k], q[1][k], q[2][k]});
      if (p_hi < q_lo || q_hi < p_lo)
        return false;
    }
    return true;
  };
  tf::generic_generate(
      candidates, crossed,
      [&](const record_t &candidate, tf::buffer<record_t> &out) {
        const auto t = corners_of(candidate[0]);
        const auto u = corners_of(candidate[1]);
        std::size_t shared = 0;
        for (const auto x : t.flats)
          for (const auto y : u.flats)
            shared += std::size_t(x == y);
        if (shared < 2 && boxes_meet(t.points, u.points) &&
            tf::exact::triangles_cross(t.points, u.points))
          out.push_back(candidate);
      },
      tf::checked);
  tbb::parallel_sort(crossed.begin(), crossed.end());
}

} // namespace tf::csg::graph
