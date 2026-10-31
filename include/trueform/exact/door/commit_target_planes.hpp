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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../canonical_plane.hpp"
#include "../meta.hpp"
#include "./find_rim_targets.hpp"
#include "./place_vertex.hpp"
#include "./placement_census.hpp"
#include "./placement_tables.hpp"
#include "./pool/pool_input_planes.hpp"
#include "./pool/wide_to_double.hpp"
#include "./quantize_direction.hpp"
#include "./round_div.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::exact::door {

/// Every face's committed plane, by ticket: `plane_of_face[f]` is a row of
/// `plane` or `-1` for a face that answers to its own exact plane. `landing`
/// is where a corner of a committed face came to stand when the face
/// committed, by the dense ticket `landing_of`.
template <typename Index, typename Int> struct committed_planes {
  tf::buffer<int> plane_of_face;
  tf::buffer<tf::exact::canonical_plane<Int>> plane;
  tf::buffer<Index> landing_of;
  tf::buffer<tf::point<Int, 3>> landing;
};

/// The most helper planes one vertex weighs; the widest-triple search is
/// cubic in the list, and a helper only chooses where a vertex lands.
inline constexpr std::size_t helper_limit = 24;

/// What a vertex RIDES: the committed planes of its own faces, which it
/// must stand on, and as helpers the exact planes of its own uncommitted
/// faces, which pass through it and may choose where on the ridden planes
/// it lands.
template <typename Index, typename Int, typename RealType, typename Owed,
          typename Helpers>
auto gather_ride_planes(const placement_tables<Index, Int, RealType> &tables,
                        const committed_planes<Index, Int> &committed,
                        std::size_t vertex, Owed &owed, Helpers &helpers)
    -> void {
  const tf::exact::canonical_plane<Int> none{};
  owed.clear();
  helpers.clear();
  for (auto k = tables.incidence_offsets[vertex];
       k < tables.incidence_offsets[vertex + 1]; ++k) {
    const auto f = std::size_t(tables.incidence[std::size_t(k)]);
    if (committed.plane_of_face[f] >= 0)
      owed.push_back(committed.plane[std::size_t(committed.plane_of_face[f])]);
    else if (tables.planes[f] != none)
      helpers.push_back(tables.planes[f]);
  }
  std::sort(owed.begin(), owed.end());
  owed.erase(std::unique(owed.begin(), owed.end()), owed.end());
  std::sort(helpers.begin(), helpers.end());
  helpers.erase(std::unique(helpers.begin(), helpers.end()), helpers.end());
  if (helpers.size() > helper_limit)
    helpers.resize(helper_limit);
}

/// THE PLANES PULL FIRST. A pool has chosen its faces' one plane and keeps
/// every face whose corners all stand on it within the band. Then every
/// face a rim vertex lands INSIDE of is given a plane the lattice is dense
/// on: the plane it already stands on when a vertex can land there, and
/// otherwise its own direction on ever finer grids, the first plane its
/// corners can follow within the band that the most of its landing
/// vertices can reach. A face no such plane serves keeps its exact plane,
/// its vertices are refused, and the census counts it.
///
/// A corner owes every plane already committed around it, so a plane is
/// laid through the landing of a corner an earlier commitment placed, and
/// the faces are decided one after another for that reason: there are as
/// many as rim vertices land inside of, which is few.
template <typename Index, typename Int, typename RealType>
auto commit_target_planes(const placement_tables<Index, Int, RealType> &tables,
                          const pool::input_pools<Int> &pools,
                          const tf::buffer<rim_target<Index>> &targets,
                          Int tolerance, committed_planes<Index, Int> &out,
                          placement_census &census) -> void {
  using plane_t = tf::exact::canonical_plane<Int>;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto n_faces = tables.planes.size();
  const auto n_vertices = tables.points.size();
  const bool pooled = pools.pool_of_face.size() == n_faces;
  out.plane.allocate(pools.plane.size());
  std::copy(pools.plane.begin(), pools.plane.end(), out.plane.begin());
  out.plane_of_face.allocate(n_faces);
  tf::parallel_for_each(
      tf::make_sequence_range(n_faces),
      [&out, &pools, pooled](std::size_t f) {
        out.plane_of_face[f] = pooled ? pools.pool_of_face[f] : -1;
      },
      tf::checked);
  out.landing_of.allocate(n_vertices);
  tf::parallel_fill(out.landing_of, Index(-1));
  out.landing.clear();
  // A VERTEX STANDS ON EVERY PLANE IT OWES OR MOVES NOT AT ALL. A pool's
  // plane is fixed by its quantization, and two dense planes meet in a line
  // the lattice may hold no point of within the band, so a corner between
  // two pools may be unable to stand on both. Such a corner keeps its input
  // point and every pool at it releases its faces there, which takes those
  // faces' other corners out of that pool's debt, so the question is asked
  // again until nothing more releases: a face still pooled then has every
  // corner exactly on its plane, and no face is left half on one.
  if (pooled && pools.plane.size() != 0) {
    struct pool_trial_t {
      tf::small_vector<plane_t, 8> owed;
      tf::small_vector<plane_t, 16> helpers;
      plane_landing<Int> landing;
    };
    tf::buffer<char> stands;
    stands.allocate(n_vertices);
    for (bool again = true; again;) {
      tf::parallel_for_each(
          tf::make_sequence_range(n_vertices),
          [&](std::size_t v, pool_trial_t &local) {
            gather_ride_planes(tables, out, v, local.owed, local.helpers);
            stands[v] = char(local.owed.empty() ||
                             place_vertex(tables.points[v], local.owed,
                                          local.helpers, tolerance,
                                          local.landing)
                                 .placed);
          },
          pool_trial_t{}, tf::checked);
      again = false;
      for (std::size_t f = 0; f < n_faces; ++f) {
        if (out.plane_of_face[f] < 0)
          continue;
        bool released = false;
        for (const auto c : tables.face_corners(f))
          released = released || !stands[std::size_t(c)];
        if (released) {
          out.plane_of_face[f] = -1;
          ++census.released_pool_faces;
          again = true;
        }
      }
    }
  }
  // the pools some face still answers to
  {
    tf::buffer<char> answered;
    answered.allocate(pools.plane.size());
    tf::parallel_fill(answered, char(0));
    for (std::size_t f = 0; f < n_faces; ++f)
      if (out.plane_of_face[f] >= 0) {
        ++census.pooled_faces;
        answered[std::size_t(out.plane_of_face[f])] = char(1);
      }
    for (const auto held : answered)
      census.pools += held ? 1 : 0;
  }
  // the faces rim vertices land inside of, each with its landing vertices,
  // in face order
  tf::buffer<std::array<Index, 2>> visits;
  for (const auto &t : targets)
    if (t.kind == rim_target_kind::face)
      visits.push_back({t.face, t.flat});
  if (visits.size() == 0)
    return;
  std::sort(visits.begin(), visits.end());
  const plane_t none{};
  struct trial_t {
    tf::small_vector<plane_t, 8> owed;
    tf::small_vector<plane_t, 16> helpers;
    plane_landing<Int> landing;
    tf::small_vector<tf::point<Int, 3>, 64> corner_at;
  } trial;
  tf::small_vector<Index, 64> corners;
  tf::small_vector<tf::point<Int, 3>, 64> chosen;
  // where a vertex stands with `candidate` added to what it rides
  const auto drop = [&](std::size_t v, const plane_t &candidate,
                        tf::point<Int, 3> &at) -> bool {
    gather_ride_planes(tables, out, v, trial.owed, trial.helpers);
    trial.owed.push_back(candidate);
    std::sort(trial.owed.begin(), trial.owed.end());
    trial.owed.erase(std::unique(trial.owed.begin(), trial.owed.end()),
                     trial.owed.end());
    const auto row = out.landing_of[v];
    if (row != Index(-1)) {
      const auto &held = out.landing[std::size_t(row)];
      bool on = true;
      for (const auto &plane : trial.owed)
        on = on && tf::exact::orient3d_plane_sign<Int>(plane, held) == 0;
      if (on) {
        at = held;
        return true;
      }
    }
    const auto placement = place_vertex(
        tables.points[v], trial.owed, trial.helpers, tolerance, trial.landing);
    at = placement.point;
    return placement.placed;
  };
  for (std::size_t first = 0; first < visits.size();) {
    const auto face = std::size_t(visits[first][0]);
    auto last = first;
    while (last < visits.size() && std::size_t(visits[last][0]) == face)
      ++last;
    const auto visit_from = first;
    first = last;
    if (out.plane_of_face[face] >= 0)
      continue;
    const auto &own = tables.planes[face];
    if (own == none) {
      ++census.target_planes_refused;
      continue;
    }
    corners.clear();
    for (const auto c : tables.face_corners(face))
      corners.push_back(c);
    std::sort(corners.begin(), corners.end());
    corners.erase(std::unique(corners.begin(), corners.end()), corners.end());
    bool decided = false;
    std::size_t best_landed = 0;
    plane_t commitment{};
    const auto consider = [&](const plane_t &candidate) {
      trial.corner_at.clear();
      for (const auto c : corners) {
        tf::point<Int, 3> at{};
        if (!drop(std::size_t(c), candidate, at))
          return;
        trial.corner_at.push_back(at);
      }
      std::size_t landed = 0;
      for (auto v = visit_from; v != last; ++v) {
        tf::point<Int, 3> at{};
        landed += drop(std::size_t(visits[v][1]), candidate, at) ? 1 : 0;
      }
      if (landed != 0 && (!decided || landed > best_landed)) {
        commitment = candidate;
        chosen = trial.corner_at;
        decided = true;
        best_landed = landed;
      }
    };
    consider(own);
    const std::array<double, 3> direction{pool::wide_to_double(own[0]),
                                          pool::wide_to_double(own[1]),
                                          pool::wide_to_double(own[2])};
    std::array<T1, 3> previous{};
    for (T1 steps = T1(2);
         best_landed < last - visit_from && steps <= T1(4) * T1(tolerance) + T1(4);
         steps = steps * T1(2)) {
      std::array<T1, 3> normal{};
      if (!quantize_direction<Int>(direction[0], direction[1], direction[2],
                                   steps, normal) ||
          normal == previous)
        continue;
      previous = normal;
      plane_t candidate{T2(normal[0]), T2(normal[1]), T2(normal[2]), T2(0)};
      // through a corner an earlier commitment already placed, else
      // through the face's corners on the whole
      Index anchor = Index(-1);
      for (const auto c : corners)
        if (out.landing_of[std::size_t(c)] != Index(-1)) {
          anchor = c;
          break;
        }
      if (anchor != Index(-1)) {
        const auto &at =
            out.landing[std::size_t(out.landing_of[std::size_t(anchor)])];
        candidate[3] = candidate[0] * T2(at[0]) + candidate[1] * T2(at[1]) +
                       candidate[2] * T2(at[2]);
      } else {
        T2 sum(0);
        for (const auto c : corners) {
          const auto &at = tables.points[std::size_t(c)];
          sum = sum + candidate[0] * T2(at[0]) + candidate[1] * T2(at[1]) +
                candidate[2] * T2(at[2]);
        }
        candidate[3] = round_div(sum, T2(corners.size()));
      }
      consider(candidate);
    }
    if (!decided) {
      ++census.target_planes_refused;
      continue;
    }
    ++census.target_planes;
    const int row = int(out.plane.size());
    out.plane.push_back(commitment);
    out.plane_of_face[face] = row;
    // a corner's landing is its whole coincidence class's
    const auto land = [&out](std::size_t v, const tf::point<Int, 3> &at) {
      auto &ticket = out.landing_of[v];
      if (ticket == Index(-1)) {
        ticket = Index(out.landing.size());
        out.landing.push_back(at);
      } else
        out.landing[std::size_t(ticket)] = at;
    };
    for (std::size_t k = 0; k < corners.size(); ++k) {
      land(std::size_t(corners[k]), chosen[k]);
      for (const auto twin : tables.twins_of(std::size_t(corners[k])))
        land(std::size_t(twin), chosen[k]);
    }
  }
}

} // namespace tf::exact::door
