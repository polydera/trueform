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
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../canonical_plane.hpp"
#include "../meta.hpp"
#include "./owed_planes.hpp"
#include "./place_vertex.hpp"
#include "./placement_census.hpp"
#include "./placement_tables.hpp"
#include "./pool/exact_lane.hpp"
#include "./pool/pool_input_planes.hpp"
#include "./pool/wide_to_double.hpp"
#include "./quantize_direction.hpp"
#include "./round_div.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::exact::door {

/// Every face's committed plane, by ticket: `plane_of_face[f]` is a row of
/// `plane` or `-1` for a face that answers to its own exact plane and is no
/// drop target. `landing` is where the corners of a committed face came to
/// stand when it committed, by the dense ticket `landing_of`; empty when
/// nothing committed beyond the pools.
template <typename Index, typename Int> struct committed_planes {
  tf::buffer<int> plane_of_face;
  tf::buffer<tf::exact::canonical_plane<Int>> plane;
  tf::buffer<Index> landing_of;
  tf::buffer<tf::point<Int, 3>> landing;
};

/// THE PLANES PULL FIRST. A pool has already chosen its faces' one plane,
/// and keeps every face whose corners all stand on it within the band. Every
/// other face a vertex of another form was admitted against gathers, with the
/// faces of that vertex lying within the band of it, into one group that
/// commits ONE plane — the plane a face of the group already stands on, and
/// otherwise the smallest-normal plane of the group's own direction — the first
/// whose corners AND visitors all stand on it within the band. A group no
/// candidate holds commits nothing: its faces keep their exact planes, are no
/// drop target, and the census counts it.
///
/// A `pinned` vertex keeps its input point: a plane commits around it only
/// where it already stands on it.
///
/// A corner owes every plane already committed around it, so a plane is
/// laid THROUGH the landing of a corner an earlier commitment already
/// placed, and two committed planes meet at a lattice point their shared
/// corner stands on. The groups are decided one after another for that
/// reason; there are as many as there are contacted faces, which is few.
template <typename Index, typename Int, typename RealType>
auto commit_contact_planes(const placement_tables<Index, Int, RealType> &tables,
                           const pool::input_pools<Int> &pools, Int tolerance,
                           const tf::buffer<char> &pinned,
                           committed_planes<Index, Int> &out,
                           placement_census &census) -> void {
  using plane_t = tf::exact::canonical_plane<Int>;
  using T1 = typename tf::exact::meta<Int>::T1;
  using coefficient_type = typename pool::exact_lane<Int>::coefficient_type;
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
  out.landing_of.clear();
  out.landing.clear();

  // a pool certified its plane on its faces' flat supports; the law asks it
  // of every corner, so a pooled face one of whose corners cannot stand on
  // all the pool planes around it does not commit
  if (pooled && pools.plane.size() != 0) {
    struct pool_trial_t {
      tf::small_vector<tf::exact::canonical_plane<Int>, 8> owed;
      plane_landing<Int> landing;
      tf::small_vector<tf::exact::canonical_plane<Int>, 1> helpers;
    };
    tf::buffer<char> holds;
    holds.allocate(n_vertices);
    tf::parallel_for_each(
        tf::make_sequence_range(n_vertices),
        [&](std::size_t v, pool_trial_t &local) {
          local.owed.clear();
          for (auto k = tables.incidence_offsets[v];
               k < tables.incidence_offsets[v + 1]; ++k) {
            const int pool = out.plane_of_face[std::size_t(
                tables.incidence[std::size_t(k)])];
            if (pool >= 0)
              local.owed.push_back(out.plane[std::size_t(pool)]);
          }
          if (local.owed.empty()) {
            holds[v] = char(1);
            return;
          }
          std::sort(local.owed.begin(), local.owed.end());
          local.owed.erase(std::unique(local.owed.begin(), local.owed.end()),
                           local.owed.end());
          if (pinned.size() != 0 && pinned[v]) {
            bool on = true;
            for (const auto &plane : local.owed)
              on = on && tf::exact::orient3d_plane_sign<Int>(
                             plane, tables.points[v]) == 0;
            holds[v] = char(on);
            return;
          }
          holds[v] = char(place_vertex(tables.points[v], local.owed,
                                       local.helpers, tolerance, local.landing)
                              .placed);
        },
        pool_trial_t{}, tf::checked);
    tf::buffer<char> released;
    released.allocate(n_faces);
    tf::parallel_for_each(
        tf::make_sequence_range(n_faces),
        [&](std::size_t f) {
          released[f] = char(0);
          if (out.plane_of_face[f] < 0)
            return;
          for (const auto c : tables.face_corners(f))
            if (!holds[std::size_t(c)])
              released[f] = char(1);
        },
        tf::checked);
    for (std::size_t f = 0; f < n_faces; ++f)
      if (released[f]) {
        out.plane_of_face[f] = -1;
        ++census.released_pool_faces;
      }
  }

  census.contacts = std::int64_t(tables.contacts.size());
  if (tables.contacts.size() == 0)
    return;

  // a face whose every corner stands within the band of a contacted face's
  // plane coincides with it at the band, and pools with it
  const plane_t none{};
  const auto near_plane = [&tables, tolerance, &none](std::size_t f,
                                                      const plane_t &plane) {
    if (plane == none)
      return false;
    const double length = std::sqrt(
        pool::wide_to_double(plane[0]) * pool::wide_to_double(plane[0]) +
        pool::wide_to_double(plane[1]) * pool::wide_to_double(plane[1]) +
        pool::wide_to_double(plane[2]) * pool::wide_to_double(plane[2]));
    const double reach = (double(tolerance) + 1.0) * length * (1.0 + 1e-9);
    for (const auto c : tables.face_corners(f))
      if (std::fabs(pool::wide_to_double(tf::exact::orient3d_plane_value<Int>(
              plane, tables.points[std::size_t(c)]))) > reach)
        return false;
    return true;
  };
  tf::buffer<std::array<Index, 2>> pairs;
  tf::generic_generate(
      tf::make_sequence_range(n_vertices), pairs,
      [&](std::size_t v, tf::buffer<std::array<Index, 2>> &into) {
        for (auto k = tables.contact_offsets[v];
             k < tables.contact_offsets[v + 1]; ++k) {
          const auto g = tables.contacts[std::size_t(k)];
          if (out.plane_of_face[std::size_t(g)] >= 0)
            continue;
          into.push_back({g, g});
          for (auto i = tables.incidence_offsets[v];
               i < tables.incidence_offsets[v + 1]; ++i) {
            const auto f = tables.incidence[std::size_t(i)];
            if (out.plane_of_face[std::size_t(f)] < 0 &&
                near_plane(std::size_t(f), tables.planes[std::size_t(g)]))
              into.push_back({f, g});
          }
        }
      });
  if (pairs.size() == 0)
    return;

  tf::buffer<Index> group_of_face;
  group_of_face.allocate(n_faces);
  const Index n_groups =
      tf::make_sparse_equivalence_class_map(pairs, group_of_face);
  const Index no_group = Index(n_faces);
  census.groups = std::int64_t(n_groups);

  // the groups' faces in face order, and their visitors, as two CSRs
  tf::buffer<Index> face_offsets, faces;
  face_offsets.allocate(std::size_t(n_groups) + 1);
  std::fill(face_offsets.begin(), face_offsets.end(), Index(0));
  for (std::size_t f = 0; f < n_faces; ++f)
    if (group_of_face[f] != no_group)
      ++face_offsets[std::size_t(group_of_face[f]) + 1];
  for (std::size_t g = 1; g < face_offsets.size(); ++g)
    face_offsets[g] += face_offsets[g - 1];
  faces.allocate(std::size_t(face_offsets[std::size_t(n_groups)]));
  {
    tf::buffer<Index> at;
    at.allocate(std::size_t(n_groups));
    std::copy(face_offsets.begin(), face_offsets.end() - 1, at.begin());
    for (std::size_t f = 0; f < n_faces; ++f)
      if (group_of_face[f] != no_group)
        faces[std::size_t(at[std::size_t(group_of_face[f])]++)] = Index(f);
  }
  tf::buffer<std::array<Index, 2>> visits;
  tf::generic_generate(
      tf::make_sequence_range(n_vertices), visits,
      [&](std::size_t v, tf::buffer<std::array<Index, 2>> &into) {
        for (auto k = tables.contact_offsets[v];
             k < tables.contact_offsets[v + 1]; ++k) {
          const auto g =
              group_of_face[std::size_t(tables.contacts[std::size_t(k)])];
          if (g != no_group)
            into.push_back({g, Index(v)});
        }
      });
  tbb::parallel_sort(visits.begin(), visits.end());
  visits.erase(std::unique(visits.begin(), visits.end()), visits.end());

  out.landing_of.allocate(n_vertices);
  tf::parallel_fill(out.landing_of, Index(-1));
  const auto group_of = [&group_of_face](std::size_t f) {
    return group_of_face[f];
  };

  struct trial_t {
    tf::small_vector<plane_t, 8> owed;
    tf::small_vector<plane_t, 16> helpers;
    plane_landing<Int> landing;
    tf::small_vector<tf::point<Int, 3>, 64> corner_at;
  } trial;
  tf::small_vector<Index, 64> corners;
  tf::small_vector<tf::point<Int, 3>, 64> chosen;
  // where a vertex stands with `candidate` added to what it already owes
  const auto drop = [&](Index group, std::size_t v, const plane_t &candidate,
                        tf::point<Int, 3> &at) -> bool {
    gather_owed_planes(tables, out.plane_of_face, out.plane, group_of, group, v,
                       true, trial.owed, trial.helpers);
    trial.owed.push_back(candidate);
    std::sort(trial.owed.begin(), trial.owed.end());
    trial.owed.erase(std::unique(trial.owed.begin(), trial.owed.end()),
                     trial.owed.end());
    if (pinned.size() != 0 && pinned[v]) {
      at = tables.points[v];
      for (const auto &plane : trial.owed)
        if (tf::exact::orient3d_plane_sign<Int>(plane, at) != 0)
          return false;
      return true;
    }
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

  auto visit = visits.begin();
  for (Index group = 0; group < n_groups; ++group) {
    const auto first = std::size_t(face_offsets[std::size_t(group)]);
    const auto last = std::size_t(face_offsets[std::size_t(group) + 1]);
    corners.clear();
    for (auto i = first; i < last; ++i)
      for (const auto c : tables.face_corners(std::size_t(faces[i])))
        corners.push_back(c);
    std::sort(corners.begin(), corners.end());
    corners.erase(std::unique(corners.begin(), corners.end()), corners.end());
    const auto visit_from = visit;
    while (visit != visits.end() && (*visit)[0] == group)
      ++visit;

    // the candidates: a plane the group's own faces stand on, then the
    // group's direction on ever finer grids
    const auto &own = tables.planes[std::size_t(faces[first])];
    bool decided = false;
    plane_t commitment{};
    const auto consider = [&](const plane_t &candidate) {
      trial.corner_at.clear();
      for (const auto c : corners) {
        tf::point<Int, 3> at{};
        if (!drop(group, std::size_t(c), candidate, at))
          return;
        trial.corner_at.push_back(at);
      }
      bool visitors = true;
      for (auto v = visit_from; v != visit && visitors; ++v) {
        tf::point<Int, 3> at{};
        visitors = drop(group, std::size_t((*v)[1]), candidate, at);
      }
      if (visitors) {
        commitment = candidate;
        chosen = trial.corner_at;
        decided = true;
      }
    };
    if (own == none) {
      ++census.uncommittable;
      continue;
    }
    consider(own);
    const std::array<double, 3> direction{pool::wide_to_double(own[0]),
                                          pool::wide_to_double(own[1]),
                                          pool::wide_to_double(own[2])};
    std::array<T1, 3> previous{};
    for (T1 steps = T1(2); !decided && steps <= T1(4) * T1(tolerance) + T1(4);
         steps = steps * T1(2)) {
      std::array<T1, 3> normal{};
      if (!quantize_direction<Int>(direction[0], direction[1], direction[2],
                                   steps, normal) ||
          normal == previous)
        continue;
      previous = normal;
      plane_t candidate{coefficient_type(normal[0]),
                        coefficient_type(normal[1]),
                        coefficient_type(normal[2]), coefficient_type(0)};
      // through a corner an earlier commitment already placed, else
      // through the group's corners on the whole
      Index anchor = Index(-1);
      for (const auto c : corners)
        if (out.landing_of[std::size_t(c)] != Index(-1)) {
          anchor = c;
          break;
        }
      if (anchor != Index(-1)) {
        const auto &at =
            out.landing[std::size_t(out.landing_of[std::size_t(anchor)])];
        candidate[3] = candidate[0] * coefficient_type(at[0]) +
                       candidate[1] * coefficient_type(at[1]) +
                       candidate[2] * coefficient_type(at[2]);
      } else {
        coefficient_type sum(0);
        for (const auto c : corners) {
          const auto &at = tables.points[std::size_t(c)];
          sum = sum + candidate[0] * coefficient_type(at[0]) +
                candidate[1] * coefficient_type(at[1]) +
                candidate[2] * coefficient_type(at[2]);
        }
        candidate[3] = round_div(sum, coefficient_type(corners.size()));
      }
      consider(candidate);
    }
    if (!decided) {
      ++census.uncommittable;
      continue;
    }
    const int row = int(out.plane.size());
    out.plane.push_back(commitment);
    for (auto i = first; i < last; ++i)
      out.plane_of_face[std::size_t(faces[i])] = row;
    for (std::size_t k = 0; k < corners.size(); ++k) {
      auto &ticket = out.landing_of[std::size_t(corners[k])];
      if (ticket == Index(-1)) {
        ticket = Index(out.landing.size());
        out.landing.push_back(chosen[k]);
      } else
        out.landing[std::size_t(ticket)] = chosen[k];
    }
    if (commitment == own)
      ++census.exact_commits;
    else
      ++census.moved_commits;
  }
}

} // namespace tf::exact::door
