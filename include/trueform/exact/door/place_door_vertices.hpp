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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../canonical_plane.hpp"
#include "../meta.hpp"
#include "./admits_placement.hpp"
#include "./commit_target_planes.hpp"
#include "./find_rim_targets.hpp"
#include "./place_vertex.hpp"
#include "./placement_census.hpp"
#include "./placement_tables.hpp"
#include "./round_div.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::exact::door {

/// One vertex the band moved, by flat id.
template <typename Index, typename Int> struct vertex_move {
  Index flat;
  tf::point<Int, 3> point;
};

/// The lattice point nearest the projection of `v` onto the edge (a, b),
/// exactly: the ends when the projection falls outside the span, else
/// `a + round((v - a).(b - a) / |b - a|^2 * (b - a))` per coordinate.
template <typename Int>
auto project_onto_edge(const tf::point<Int, 3> &v, const tf::point<Int, 3> &a,
                       const tf::point<Int, 3> &b)
    -> std::array<typename tf::exact::meta<Int>::T1, 3> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  std::array<T1, 3> d{}, w{};
  for (std::size_t k = 0; k < 3; ++k) {
    d[k] = T1(b[k]) - T1(a[k]);
    w[k] = T1(v[k]) - T1(a[k]);
  }
  const T2 dot = T2(d[0]) * T2(w[0]) + T2(d[1]) * T2(w[1]) + T2(d[2]) * T2(w[2]);
  const T2 length2 =
      T2(d[0]) * T2(d[0]) + T2(d[1]) * T2(d[1]) + T2(d[2]) * T2(d[2]);
  std::array<T1, 3> out{};
  if (length2 == T2(0) || dot <= T2(0)) {
    for (std::size_t k = 0; k < 3; ++k)
      out[k] = T1(a[k]);
    return out;
  }
  if (dot >= length2) {
    for (std::size_t k = 0; k < 3; ++k)
      out[k] = T1(b[k]);
    return out;
  }
  for (std::size_t k = 0; k < 3; ++k)
    out[k] = T1(a[k]) +
             T1(round_div(T3(dot) * T3(d[k]), T3(length2)));
  return out;
}

/// THEN THE VERTICES LAND. First every vertex STANDS: on the committed
/// planes of its own faces, which it rides, or where the input put it. Then
/// a rim vertex with a target lands on the one feature it targets, read
/// where that feature stands. A corner is the corner's own stand; two
/// corners that target each other are one weld, the smaller flat id its
/// anchor, and a corner that targets a corner is followed to the anchor
/// its chain ends at, so no landing ever reads a point that moves. An edge
/// is the lattice point nearest the vertex's projection onto it, and the
/// vertex is stated on the edge by name (`edge_names`, one row `{vertex,
/// face, slot}`) so the exact tier reads it there whatever the rounding
/// left; a face is a lattice point on the face's committed plane. Every
/// landing stands on what the vertex rides and within the band of its input
/// point, or the vertex keeps its stand and the refusal is counted by
/// feature.
///
/// The product is sparse: `moved` holds the vertices that moved, in flat
/// order, and `ticket` maps a flat id to its row or `-1`. Both stay empty
/// when nothing moved, which is the band answering exactly as zero does.
template <typename Index, typename Int, typename RealType>
auto place_door_vertices(const placement_tables<Index, Int, RealType> &tables,
                         const committed_planes<Index, Int> &committed,
                         const tf::buffer<rim_target<Index>> &targets,
                         Int tolerance, tf::buffer<Index> &ticket,
                         tf::buffer<tf::point<Int, 3>> &moved,
                         tf::buffer<std::array<Index, 3>> &edge_names,
                         placement_census &census) -> void {
  using plane_t = tf::exact::canonical_plane<Int>;
  using T1 = typename tf::exact::meta<Int>::T1;
  const auto n_vertices = tables.points.size();
  struct local_t {
    tf::buffer<vertex_move<Index, Int>> moves;
    tf::buffer<std::array<Index, 3>> names;
    placement_census census{};
    tf::small_vector<plane_t, 8> owed;
    tf::small_vector<plane_t, 16> helpers;
    plane_landing<Int> landing;
  };
  struct result_t {
    tf::buffer<vertex_move<Index, Int>> moves;
    tf::buffer<std::array<Index, 3>> names;
    placement_census census{};
  } result;
  const auto stands_on = [](const auto &owed, const tf::point<Int, 3> &at) {
    for (const auto &plane : owed)
      if (tf::exact::orient3d_plane_sign<Int>(plane, at) != 0)
        return false;
    return true;
  };
  const auto admits = [tolerance](const tf::point<Int, 3> &original,
                                  const tf::point<Int, 3> &at) {
    return admits_placement<Int>(
        original, std::array<T1, 3>{T1(at[0]), T1(at[1]), T1(at[2])},
        tolerance);
  };
  // THE STAND: where every vertex is before any landing, one disjoint
  // store per vertex
  tf::buffer<tf::point<Int, 3>> stand;
  stand.allocate(n_vertices);
  struct ride_t {
    tf::small_vector<plane_t, 8> owed;
    tf::small_vector<plane_t, 16> helpers;
    plane_landing<Int> landing;
    std::int64_t refused = 0;
  };
  std::int64_t ride_refused = 0;
  tf::blocked_reduce(
      tf::make_sequence_range(n_vertices), ride_refused, ride_t{},
      [&](const auto &block, ride_t &local) {
        for (const auto v : block) {
          const auto &original = tables.points[v];
          stand[v] = original;
          gather_ride_planes(tables, committed, std::size_t(v), local.owed,
                             local.helpers);
          if (local.owed.empty())
            continue;
          const auto row = committed.landing_of[v];
          if (row != Index(-1) &&
              stands_on(local.owed, committed.landing[std::size_t(row)])) {
            stand[v] = committed.landing[std::size_t(row)];
            continue;
          }
          const auto placement = place_vertex(
              original, local.owed, local.helpers, tolerance, local.landing);
          if (placement.placed)
            stand[v] = placement.point;
          else
            ++local.refused;
        }
      },
      [](const ride_t &local, std::int64_t &into) { into += local.refused; },
      tf::checked);
  const auto target_of = [&targets](Index v) -> const rim_target<Index> * {
    const auto at = std::lower_bound(
        targets.begin(), targets.end(), v,
        [](const rim_target<Index> &t, Index flat) { return t.flat < flat; });
    return at != targets.end() && at->flat == v ? &*at : nullptr;
  };
  // the anchor a corner chain ends at: a corner no corner targets, or the
  // smaller of two that target each other; a chain the band cannot close
  // in a few steps anchors nowhere
  const auto anchor_of = [&target_of](Index v) -> Index {
    for (int step = 0; step < 8; ++step) {
      const auto *t = target_of(v);
      if (t == nullptr || t->kind != rim_target_kind::corner)
        return v;
      const auto *back = target_of(t->feature);
      if (back != nullptr && back->kind == rim_target_kind::corner &&
          back->feature == v)
        return v < t->feature ? v : t->feature;
      v = t->feature;
    }
    return Index(-1);
  };
  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(n_vertices), result, local_t{},
      [&](const auto &block, local_t &local) {
        local.moves.clear();
        local.names.clear();
        local.census = placement_census{};
        auto target = std::lower_bound(
            targets.begin(), targets.end(), Index(*block.begin()),
            [](const rim_target<Index> &t, Index v) { return t.flat < v; });
        for (const auto v : block) {
          const auto &original = tables.points[v];
          tf::point<Int, 3> at = stand[v];
          while (target != targets.end() && target->flat < Index(v))
            ++target;
          // a landing is the whole coincidence class's: the class lands
          // where its least member with a target lands
          const rim_target<Index> *landing = nullptr;
          if (target != targets.end() && target->flat == Index(v))
            landing = &*target;
          for (const auto twin : tables.twins_of(v))
            if (landing == nullptr || twin < landing->flat)
              if (const auto *theirs = target_of(twin))
                landing = theirs;
          if (landing != nullptr) {
            const auto &t = *landing;
            gather_ride_planes(tables, committed, std::size_t(v), local.owed,
                               local.helpers);
            switch (t.kind) {
            case rim_target_kind::corner: {
              ++local.census.corner_targets;
              const auto anchor = anchor_of(t.flat);
              if (anchor == t.flat)
                break;
              if (anchor == Index(-1)) {
                ++local.census.corner_refused;
                break;
              }
              const auto &c = stand[std::size_t(anchor)];
              if (stands_on(local.owed, c) && admits(original, c))
                at = c;
              else
                ++local.census.corner_refused;
              break;
            }
            case rim_target_kind::edge: {
              ++local.census.edge_targets;
              const auto corners = tables.face_corners(std::size_t(t.face));
              const auto n = corners.size();
              const auto a = corners[std::size_t(t.feature)];
              const auto b = corners[(std::size_t(t.feature) + 1) % n];
              const auto on = project_onto_edge<Int>(
                  original, stand[std::size_t(a)], stand[std::size_t(b)]);
              const tf::point<Int, 3> q{Int(on[0]), Int(on[1]), Int(on[2])};
              if (admits_placement<Int>(original, on, tolerance) &&
                  stands_on(local.owed, q)) {
                at = q;
                local.names.push_back({Index(v), t.face, t.feature});
              } else
                ++local.census.edge_refused;
              break;
            }
            case rim_target_kind::face: {
              ++local.census.face_targets;
              const auto plane = committed.plane_of_face[std::size_t(t.face)];
              if (plane < 0) {
                ++local.census.face_refused;
                break;
              }
              local.owed.push_back(committed.plane[std::size_t(plane)]);
              std::sort(local.owed.begin(), local.owed.end());
              local.owed.erase(
                  std::unique(local.owed.begin(), local.owed.end()),
                  local.owed.end());
              const auto row = committed.landing_of[v];
              if (row != Index(-1) &&
                  stands_on(local.owed, committed.landing[std::size_t(row)]))
                at = committed.landing[std::size_t(row)];
              else {
                const auto placement =
                    place_vertex(original, local.owed, local.helpers,
                                 tolerance, local.landing);
                if (placement.placed)
                  at = placement.point;
                else
                  ++local.census.face_refused;
              }
              break;
            }
            case rim_target_kind::on:
              break;
            }
          }
          if (at != original) {
            ++local.census.moved;
            local.moves.push_back({Index(v), at});
          }
        }
      },
      [](const local_t &local, result_t &into) {
        const auto from = into.moves.size();
        into.moves.reallocate(from + local.moves.size());
        std::copy(local.moves.begin(), local.moves.end(),
                  into.moves.begin() + from);
        const auto names_from = into.names.size();
        into.names.reallocate(names_from + local.names.size());
        std::copy(local.names.begin(), local.names.end(),
                  into.names.begin() + names_from);
        into.census.moved += local.census.moved;
        into.census.corner_targets += local.census.corner_targets;
        into.census.corner_refused += local.census.corner_refused;
        into.census.edge_targets += local.census.edge_targets;
        into.census.edge_refused += local.census.edge_refused;
        into.census.face_targets += local.census.face_targets;
        into.census.face_refused += local.census.face_refused;
      });
  result.census.ride_refused = ride_refused;
  census.moved = result.census.moved;
  census.ride_refused = result.census.ride_refused;
  census.corner_targets = result.census.corner_targets;
  census.corner_refused = result.census.corner_refused;
  census.edge_targets = result.census.edge_targets;
  census.edge_refused = result.census.edge_refused;
  census.face_targets = result.census.face_targets;
  census.face_refused = result.census.face_refused;
  edge_names = std::move(result.names);
  const auto &moves = result.moves;
  ticket.clear();
  moved.allocate(moves.size());
  if (moves.size() == 0)
    return;
  ticket.allocate(n_vertices);
  tf::parallel_fill(ticket, Index(-1));
  tf::parallel_for_each(
      tf::make_sequence_range(moves.size()),
      [&ticket, &moved, &moves](std::size_t r) {
        ticket[std::size_t(moves[r].flat)] = Index(r);
        moved[r] = moves[r].point;
      },
      tf::checked);
}

} // namespace tf::exact::door
