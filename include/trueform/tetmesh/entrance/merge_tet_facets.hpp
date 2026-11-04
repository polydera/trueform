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
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/name_plane_carriers.hpp"
#include "../../intersect/graph/plane_face_support.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/manifold_edge_link.hpp"
#include "../../topology/manifold_edge_peer.hpp"
#include "./count_tet_open_edges.hpp"
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
/// their first triangle; returns the number of facets. Every triangle names
/// the plane it stands on by the house's plane identity
/// (@ref tf::intersect::graph::name_plane_carriers), and, the name being
/// unoriented, states which way its own normal faces the name's. Two
/// triangles are one facet when they share an edge no third triangle
/// holds, run it in opposite directions, wear one name that is not void
/// and face it alike — so they lie on either side of the edge, wound
/// alike; facets are the classes those joins close. A fold, two coplanar
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
/// triangles thrown out as edges, as lines, and those the subdivision added,
/// and `open_edges` the edges `kept` leaves open (@ref tet_open_edges), read
/// off the link's peers when nothing was thrown out.
template <typename Int, typename Index, typename Points, typename Triangles,
          typename Membership, typename Link>
auto merge_tet_facets(const Points &points, const Triangles &triangles,
                      const Membership &membership, const Link &link,
                      tf::buffer<std::array<Index, 3>> &kept,
                      tf::buffer<Index> &origin, tf::buffer<Index> &facet_of,
                      std::array<std::size_t, 3> &thrown,
                      tet_open_edges &open_edges) -> Index {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using peer_t = tf::manifold_edge_peer<Index>;
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
  tf::buffer<tf::exact::canonical_plane<Int>> names;
  tf::buffer<signed char> facing;
  tf::buffer<signed char> kind;
  names.allocate(n);
  facing.allocate(n);
  kind.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&names, &facing, &kind, &points, &triangles, &corners](std::size_t t) {
        const auto &triangle = triangles[t];
        const auto support = tf::intersect::graph::plane_face_support<Int>(
            triangle, [&points, &triangle](std::size_t corner) {
              const auto &p = points[std::size_t(triangle[corner])];
              return tf::exact::pt3<Int>{Int(p[0]), Int(p[1]), Int(p[2])};
            });
        names[t] = tf::exact::make_canonical_plane<Int>(support);
        signed char sign = 0;
        for (std::size_t k = 0; k < 3 && !sign; ++k)
          sign = support.normal[k] > T2(0)   ? 1
                 : support.normal[k] < T2(0) ? -1
                                             : 0;
        facing[t] = sign;
        const auto c = corners(t);
        kind[t] = !tf::intersect::graph::is_plane_name_void<Int>(names[t])
                      ? whole
                  : c[0] == c[1] || c[1] == c[2] || c[0] == c[2] ? edge
                                                                 : line;
      },
      tf::checked(64));
  tf::buffer<std::pair<Index, Index>> joins;
  tf::sequenced_generate(
      tf::make_sequence_range(n), joins,
      [&link, &names, &facing, &corners, &runs](std::size_t t, auto &out) {
        const auto &name = names[t];
        if (tf::intersect::graph::is_plane_name_void<Int>(name))
          return;
        const auto c = corners(t);
        const auto peers = link[t];
        for (std::size_t k = 0; k < 3; ++k) {
          const Index p = peers[k].face_peer;
          if (p < 0 || std::size_t(p) <= t)
            continue;
          const Index u = c[k], v = c[k == 2 ? 0 : k + 1];
          if (!runs(std::size_t(p), v, u))
            continue;
          if (name == names[std::size_t(p)] &&
              facing[t] == facing[std::size_t(p)])
            out.push_back({Index(t), p});
        }
      },
      tf::checked);
  tf::buffer<Index> classes;
  classes.allocate(n);
  const auto n_classes = tf::make_dense_equivalence_class_map(joins, classes);
  thrown = {0, 0, 0};
  for (std::size_t t = 0; t < n; ++t) {
    thrown[0] += kind[t] == edge;
    thrown[1] += kind[t] == line;
  }
  kept.clear();
  origin.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(n), std::tie(kept, origin),
      [&](std::size_t t, auto &out) {
        if (kind[t] != whole)
          return;
        auto &faces = std::get<0>(out);
        auto &from = std::get<1>(out);
        const auto c = corners(t);
        if (!thrown[1]) {
          faces.push_back(c);
          from.push_back(Index(t));
          return;
        }
        tf::small_vector<Index, 16> polygon;
        std::array<std::size_t, 4> start{};
        tf::small_vector<Index, 8> on_edge;
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
          start[k] = polygon.size();
          polygon.push_back(u);
          tf::small_vector<Index, 16> chain;
          lines_on(t, u, v, chain);
          if (!chain.size())
            continue;
          const auto pu = at(u), pv = at(v);
          T2 length = 0;
          for (std::size_t axis = 0; axis < 3; ++axis)
            length += T2(T1(pv[axis]) - pu[axis]) * (T1(pv[axis]) - pu[axis]);
          tf::small_vector<std::pair<T2, Index>, 16> inside;
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
          std::sort(inside.begin(), inside.end());
          inside.erase(std::unique(inside.begin(), inside.end()), inside.end());
          for (const auto &entry : inside)
            polygon.push_back(entry.second);
        }
        start[3] = polygon.size();
        if (polygon.size() == 3) {
          faces.push_back(c);
          from.push_back(Index(t));
          return;
        }
        const auto emit = [&faces, &from, t](Index a, Index b, Index apex) {
          faces.push_back({a, b, apex});
          from.push_back(Index(t));
        };
        const auto fan = [&polygon, &start, &emit](std::size_t edge, Index apex,
                                                   Index first, Index last) {
          Index previous = first;
          for (std::size_t i = start[edge] + 1; i < start[edge + 1]; ++i) {
            emit(previous, polygon[i], apex);
            previous = polygon[i];
          }
          emit(previous, last, apex);
        };
        std::size_t k = 0;
        while (start[k + 1] - start[k] == 1)
          ++k;
        const Index u = c[k], v = c[(k + 1) % 3], o = c[(k + 2) % 3];
        const std::size_t before = (k + 2) % 3, after = (k + 1) % 3;
        const Index head = polygon[start[k] + 1],
                    tail = polygon[start[k + 1] - 1];
        if (start[before + 1] - start[before] > 1)
          fan(before, head, o, u);
        else
          emit(u, head, o);
        for (std::size_t i = start[k] + 1; i + 1 < start[k + 1]; ++i)
          emit(polygon[i], polygon[i + 1], o);
        if (start[after + 1] - start[after] > 1)
          fan(after, tail, v, o);
        else
          emit(tail, v, o);
      },
      tf::checked);
  thrown[2] = kept.size() - (n - thrown[0] - thrown[1]);
  if (thrown[0] || thrown[1])
    open_edges = count_tet_open_edges(kept);
  else
    open_edges = tf::reduce(
        tf::make_sequence_range(n),
        [&link, &membership, &triangles, &corners, &runs](tet_open_edges open,
                                                          const auto &at) {
          if constexpr (std::is_same_v<std::decay_t<decltype(at)>,
                                       tet_open_edges>) {
            open.unbalanced += at.unbalanced;
            open.boundary += at.boundary;
            return open;
          } else {
            const std::size_t t = std::size_t(at);
            const auto c = corners(t);
            const auto peers = link[t];
            for (std::size_t k = 0; k < 3; ++k) {
              const Index p = peers[k].face_peer;
              const Index u = c[k], v = c[k == 2 ? 0 : k + 1];
              if (p == peer_t::boundary) {
                ++open.unbalanced;
                ++open.boundary;
                continue;
              }
              if (p >= 0) {
                open.unbalanced +=
                    std::size_t(p) > t && runs(std::size_t(p), u, v);
                continue;
              }
              if (p != peer_t::non_manifold_representative)
                continue;
              int balance = 1;
              tf::face_edge_neighbors_apply(
                  membership, triangles, Index(t), u, v, [&](Index r) {
                    balance += runs(std::size_t(r), u, v) ? 1 : -1;
                    return false;
                  });
              open.unbalanced += balance != 0;
            }
            return open;
          }
        },
        tet_open_edges{});
  facet_of.allocate(kept.size());
  if (!thrown[0] && !thrown[1]) {
    std::copy(classes.begin(), classes.end(), facet_of.begin());
    return Index(n_classes);
  }
  tf::buffer<Index> renumber;
  renumber.allocate_and_initialize(std::size_t(n_classes), Index(-1));
  Index facets = 0;
  for (std::size_t k = 0; k < kept.size(); ++k) {
    auto &facet = renumber[std::size_t(classes[std::size_t(origin[k])])];
    if (facet == Index(-1))
      facet = facets++;
    facet_of[k] = facet;
  }
  return facets;
}

} // namespace tf::tetmesh
