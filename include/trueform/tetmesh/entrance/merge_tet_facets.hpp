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
#include "../../core/algorithm/compact_ids_and_make_map.hpp"
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/plane_face_support.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/manifold_edge_link.hpp"
#include "../../topology/manifold_edge_peer.hpp"
#include "./tet_facet_workspace.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::tetmesh {

/// Merge the triangles `triangles` over the lattice points `points` into
/// facets, their neighbours read from `link`, the surface's manifold edge
/// link, and every triangle on an edge from `membership`, its face
/// membership: `kept` holds the triangles the facets are made of, `origin`
/// the input triangle each comes from and `facet_of` its facet, numbered by
/// their first triangle; returns the number of facets. Every triangle stands
/// on the plane its corners support (@ref tf::exact::plane_support), void
/// where they stand on one line, and states which way its own normal faces.
/// Two triangles are one facet when they share an edge no third triangle
/// holds, run it in opposite directions, stand on one plane that is not
/// void and face it alike — so they lie on either side of the edge, wound
/// alike; facets are the classes those joins close. Two triangles sharing
/// an edge wear one plane name (@ref tf::intersect::graph::name_plane_carriers)
/// exactly when the one's far corner stands on the other's plane, so a join
/// reads that orient3d and no name is formed. A fold, two coplanar
/// triangles wound against each other, is two facets, and so is a plane two
/// runs of triangles reach without touching.
///
/// A triangle whose name is void is thrown out where the merge meets it: one
/// with a repeated corner is an edge, one whose three corners stand on one
/// line is a line. A line's corners lie on the edges it shares, so a kept
/// triangle on such an edge is subdivided at the corners strictly inside it,
/// found through the chain of void triangles sharing edges, and re-emitted
/// as fans in its own winding, as the iso cut fans a convex chain
/// (@ref tf::iso::split_iso_face_chord): from the corner opposite its first
/// split edge, the fan's end triangles fanning again from their own corner
/// off that edge over the other edges' corners, so the surface stays
/// closed; its pieces keep its facet. `thrown` counts the
/// triangles thrown out as edges, as lines, and those the subdivision added.
/// The merge walks on the tables of `work`.
template <typename Int, typename Index, typename Points, typename Triangles,
          typename Membership, typename Link>
auto merge_tet_facets(const Points &points, const Triangles &triangles,
                      const Membership &membership, const Link &link,
                      tet_facet_workspace<Index, Int> &work,
                      tf::buffer<std::array<Index, 3>> &kept,
                      tf::buffer<Index> &origin, tf::buffer<Index> &facet_of,
                      std::array<std::size_t, 3> &thrown) -> Index {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  enum : signed char { whole = 0, edge = 1, line = 2 };
  const std::size_t n = triangles.size();
  const auto at = [&points](Index p) {
    const auto &q = points[std::size_t(p)];
    return tf::exact::pt3<Int>{Int(q[0]), Int(q[1]), Int(q[2])};
  };
  const auto corners = [&triangles](std::size_t t) {
    const auto &triangle = triangles[t];
    return std::array<Index, 3>{Index(triangle[0]), Index(triangle[1]),
                                Index(triangle[2])};
  };
  const auto runs = [&corners](std::size_t t, Index u, Index v) {
    const auto c = corners(t);
    for (std::size_t k = 0; k < 3; ++k)
      if (c[k] == u && c[k == 2 ? 0 : k + 1] == v)
        return true;
    return false;
  };
  auto &facing = work.facing;
  auto &kind = work.kind;
  facing.allocate(n);
  kind.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&facing, &kind, &points, &triangles, &corners](std::size_t t) {
        const auto &triangle = triangles[t];
        const auto support = tf::intersect::graph::plane_face_support<Int>(
            triangle, [&points, &triangle](std::size_t corner) {
              const auto &p = points[std::size_t(triangle[corner])];
              return tf::exact::pt3<Int>{Int(p[0]), Int(p[1]), Int(p[2])};
            });
        signed char sign = 0;
        for (std::size_t k = 0; k < 3 && !sign; ++k)
          sign = support.normal[k] > T2(0)   ? 1
                 : support.normal[k] < T2(0) ? -1
                                             : 0;
        facing[t] = sign;
        const auto c = corners(t);
        kind[t] = sign                                           ? whole
                  : c[0] == c[1] || c[1] == c[2] || c[0] == c[2] ? edge
                                                                 : line;
      },
      tf::checked(64));
  auto &joins = work.joins;
  joins.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(n), joins,
      [&link, &kind, &facing, &corners, &runs, &at](std::size_t t, auto &out) {
        if (kind[t] != whole)
          return;
        const auto c = corners(t);
        const auto peers = link(t);
        for (std::size_t k = 0; k < 3; ++k) {
          const Index p = peers[k].face_peer;
          if (p < 0 || std::size_t(p) <= t)
            continue;
          const Index u = c[k], v = c[k == 2 ? 0 : k + 1];
          if (!runs(std::size_t(p), v, u))
            continue;
          if (kind[std::size_t(p)] != whole ||
              facing[t] != facing[std::size_t(p)])
            continue;
          const auto q = corners(std::size_t(p));
          Index apex = q[0];
          for (std::size_t j = 0; j < 3; ++j)
            if (q[j] != u && q[j] != v)
              apex = q[j];
          if (!tf::exact::orient3d_sign(at(c[0]), at(c[1]), at(c[2]), at(apex)))
            out.push_back({Index(t), p});
        }
      },
      tf::checked);
  auto &classes = work.classes;
  classes.allocate(n);
  const auto n_classes = tf::make_dense_equivalence_class_map(joins, classes);
  const auto counted = tf::reduce(
      tf::make_sequence_range(n),
      [&kind](std::array<std::size_t, 2> sum, const auto &at) {
        if constexpr (std::is_same_v<std::decay_t<decltype(at)>,
                                     std::array<std::size_t, 2>>) {
          sum[0] += at[0];
          sum[1] += at[1];
        } else {
          sum[0] += kind[std::size_t(at)] == edge;
          sum[1] += kind[std::size_t(at)] == line;
        }
        return sum;
      },
      std::array<std::size_t, 2>{0, 0}, tf::checked);
  thrown = {counted[0], counted[1], 0};
  kept.clear();
  origin.clear();
  // A corner a line triangle leaves strictly inside one edge, at its
  // projection along that edge.
  struct edge_corner {
    T2 along;
    Index corner;
  };
  // The boundary chain of one merged facet, the line triangles it is walked
  // through, and the corners they leave on one of its edges.
  struct local_t {
    tf::buffer<Index> ring;
    tf::buffer<Index> on_edge;
    tf::buffer<Index> chain;
    tf::buffer<edge_corner> inside;
  };
  tf::sequenced_generate(
      tf::make_sequence_range(n), std::tie(kept, origin),
      [&](std::size_t t, auto &out, local_t &local) {
        if (kind[t] != whole)
          return;
        auto &emitted = std::get<0>(out);
        auto &from = std::get<1>(out);
        const auto c = corners(t);
        if (!thrown[1]) {
          emitted.push_back(c);
          from.push_back(Index(t));
          return;
        }
        auto &ring = local.ring;
        auto &on_edge = local.on_edge;
        ring.clear();
        std::array<std::size_t, 4> start{};
        const auto lines_on = [&](std::size_t face, Index u, Index v,
                                  auto &chain) {
          on_edge.clear();
          tf::face_edge_neighbors(membership, triangles, Index(face), u, v,
                                  std::back_inserter(on_edge));
          for (auto r : on_edge)
            if (kind[std::size_t(r)] == line &&
                std::find(chain.begin(), chain.end(), r) == chain.end())
              chain.push_back(r);
        };
        for (std::size_t k = 0; k < 3; ++k) {
          const Index u = c[k], v = c[(k + 1) % 3];
          start[k] = ring.size();
          ring.push_back(u);
          auto &chain = local.chain;
          chain.clear();
          lines_on(t, u, v, chain);
          if (!chain.size())
            continue;
          const auto pu = at(u), pv = at(v);
          T2 length = 0;
          for (std::size_t axis = 0; axis < 3; ++axis)
            length += T2(T1(pv[axis]) - pu[axis]) * (T1(pv[axis]) - pu[axis]);
          auto &inside = local.inside;
          inside.clear();
          for (std::size_t head = 0; head < chain.size(); ++head) {
            const auto l = corners(std::size_t(chain[head]));
            for (std::size_t j = 0; j < 3; ++j) {
              const auto pp = at(l[j]);
              T2 along = 0;
              for (std::size_t axis = 0; axis < 3; ++axis)
                along +=
                    T2(T1(pp[axis]) - pu[axis]) * (T1(pv[axis]) - pu[axis]);
              if (l[j] != u && l[j] != v && along > T2(0) && along < length)
                inside.push_back({along, l[j]});
              lines_on(std::size_t(chain[head]), l[j], l[(j + 1) % 3], chain);
            }
          }
          std::sort(inside.begin(), inside.end(),
                    [](const edge_corner &x, const edge_corner &y) {
                      return x.along < y.along ||
                             (x.along == y.along && x.corner < y.corner);
                    });
          inside.erase_till_end(
              std::unique(inside.begin(), inside.end(),
                          [](const edge_corner &x, const edge_corner &y) {
                            return x.along == y.along && x.corner == y.corner;
                          }));
          for (const auto &entry : inside)
            ring.push_back(entry.corner);
        }
        start[3] = ring.size();
        if (ring.size() == 3) {
          emitted.push_back(c);
          from.push_back(Index(t));
          return;
        }
        const auto emit = [&emitted, &from, t](Index a, Index b, Index apex) {
          emitted.push_back({a, b, apex});
          from.push_back(Index(t));
        };
        const auto fan = [&ring, &start, &emit](std::size_t edge, Index apex,
                                                Index first, Index last) {
          Index previous = first;
          for (std::size_t i = start[edge] + 1; i < start[edge + 1]; ++i) {
            emit(previous, ring[i], apex);
            previous = ring[i];
          }
          emit(previous, last, apex);
        };
        std::size_t k = 0;
        while (start[k + 1] - start[k] == 1) {
          ++k;
          if constexpr (tf::core::verification)
            tf::core::verify(k < 3, "a split triangle has a side carrying a "
                                    "point");
        }
        const Index u = c[k], v = c[(k + 1) % 3], o = c[(k + 2) % 3];
        const std::size_t before = (k + 2) % 3, after = (k + 1) % 3;
        const Index head = ring[start[k] + 1], tail = ring[start[k + 1] - 1];
        if (start[before + 1] - start[before] > 1)
          fan(before, head, o, u);
        else
          emit(u, head, o);
        for (std::size_t i = start[k] + 1; i + 1 < start[k + 1]; ++i)
          emit(ring[i], ring[i + 1], o);
        if (start[after + 1] - start[after] > 1)
          fan(after, tail, v, o);
        else
          emit(tail, v, o);
      },
      local_t{}, tf::checked);
  thrown[2] = kept.size() - (n - thrown[0] - thrown[1]);
  facet_of.allocate(kept.size());
  if (!thrown[0] && !thrown[1]) {
    tf::parallel_copy(classes, facet_of);
    return Index(n_classes);
  }
  tf::parallel_for_each(
      tf::make_sequence_range(kept.size()),
      [&facet_of, &classes, &origin](std::size_t k) {
        facet_of[k] = classes[std::size_t(origin[k])];
      },
      tf::checked);
  auto &renumber = work.renumber;
  renumber.allocate(std::size_t(n_classes));
  return tf::compact_ids_and_make_map(tf::make_range(facet_of),
                                      tf::make_range(renumber));
}

} // namespace tf::tetmesh
