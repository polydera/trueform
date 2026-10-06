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
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/face_membership_like.hpp"
#include "./hole_metric.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// The neighbourhood one patch is faired over: its own triangles, the plan of
/// every carrying face those triangles reach, and every triangle host face
/// standing at one of its rim vertices — all deduplicated, numbered by
/// ascending flat identity and canonically ordered.
///
/// `from_patch` says which of the triangles are the patch's own, and
/// `free_ids` names the vertices the solve may move: the points the patch
/// minted strictly inside itself. `free_of` is their back-map, `-1` for a
/// vertex the solve holds fixed.
///
/// @tparam Index The index type of the mesh the stencil is stated against.
template <typename Index> struct hole_fairing_stencil {
  tf::buffer<Index> names;
  tf::buffer<tf::point<double, 3>> positions;
  tf::buffer<std::array<Index, 3>> triangles;
  tf::buffer<char> from_patch;
  tf::buffer<Index> free_ids;
  tf::buffer<Index> free_of;
};

/// The scratch one stencil's assembly walks on.
template <typename Index> struct hole_fairing_stencil_scratch {
  tf::buffer<std::array<Index, 4>> records;
  tf::buffer<std::array<Index, 2>> sides;
  tf::buffer<Index> boundary;
  tf::buffer<Index> interior;
  tf::buffer<Index> hosts;
};

/// Gather the stencil of one group's patch.
///
/// Returns whether the neighbourhood is COMPLETE. A rim vertex whose host fan
/// holds a face that is not a triangle has a row this tier cannot state, so
/// that face is left out and the stencil states it is incomplete — enough for
/// the seam the group publishes, never enough to solve on.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Triangles, typename RealT>
auto build_hole_fairing_stencil(
    const tf::polygons<Policy> &polygons,
    const tf::face_membership_like<MembershipPolicy> &fm, Index n_points,
    const Triangles &patch, const tf::buffer<Index> &plan_faces,
    const tf::offset_block_buffer<Index, std::array<Index, 3>> &plan_triangles,
    const tf::buffer<tf::point<RealT, 3>> &minted,
    tf::fill::hole_fairing_stencil_scratch<Index> &scratch,
    tf::fill::hole_fairing_stencil<Index> &stencil) -> bool {
  const auto mesh_faces = polygons.faces();
  const auto mesh_points = polygons.points();

  scratch.records.clear();
  scratch.sides.clear();
  scratch.interior.clear();
  for (const auto &triangle : patch) {
    scratch.records.push_back(
        {triangle[0], triangle[1], triangle[2], Index(1)});
    for (int side = 0; side < 3; ++side) {
      const Index a = triangle[std::size_t(side)];
      const Index b = triangle[std::size_t((side + 1) % 3)];
      scratch.sides.push_back({std::min(a, b), std::max(a, b)});
      scratch.interior.push_back(a);
    }
  }
  std::sort(scratch.sides.begin(), scratch.sides.end());
  std::sort(scratch.interior.begin(), scratch.interior.end());
  scratch.interior.erase_till_end(
      std::unique(scratch.interior.begin(), scratch.interior.end()));

  scratch.boundary.clear();
  for (std::size_t k = 0; k < scratch.sides.size();) {
    std::size_t run = k;
    while (run < scratch.sides.size() && scratch.sides[run] == scratch.sides[k])
      ++run;
    if (run == k + 1) {
      scratch.boundary.push_back(scratch.sides[k][0]);
      scratch.boundary.push_back(scratch.sides[k][1]);
    }
    k = run;
  }
  std::sort(scratch.boundary.begin(), scratch.boundary.end());
  scratch.boundary.erase_till_end(
      std::unique(scratch.boundary.begin(), scratch.boundary.end()));

  scratch.hosts.clear();
  for (auto vertex : scratch.boundary) {
    if (vertex >= n_points)
      continue;
    for (auto face : fm[std::size_t(vertex)])
      scratch.hosts.push_back(face);
  }
  std::sort(scratch.hosts.begin(), scratch.hosts.end());
  scratch.hosts.erase_till_end(
      std::unique(scratch.hosts.begin(), scratch.hosts.end()));

  bool complete = true;
  for (auto face : scratch.hosts) {
    const auto plan = std::lower_bound(plan_faces.begin(), plan_faces.end(),
                                       face);
    if (plan != plan_faces.end() && *plan == face) {
      const auto block =
          plan_triangles[std::size_t(plan - plan_faces.begin())];
      for (const auto &triangle : block)
        scratch.records.push_back(
            {triangle[0], triangle[1], triangle[2], Index(0)});
      continue;
    }
    const auto corners = mesh_faces[std::size_t(face)];
    if (corners.size() != 3) {
      complete = false;
      continue;
    }
    scratch.records.push_back(
        {Index(corners[0]), Index(corners[1]), Index(corners[2]), Index(0)});
  }

  stencil.names.clear();
  for (const auto &record : scratch.records)
    for (int corner = 0; corner < 3; ++corner)
      stencil.names.push_back(record[std::size_t(corner)]);
  std::sort(stencil.names.begin(), stencil.names.end());
  stencil.names.erase_till_end(
      std::unique(stencil.names.begin(), stencil.names.end()));
  const auto local_of = [&stencil](Index name) {
    return Index(std::lower_bound(stencil.names.begin(), stencil.names.end(),
                                  name) -
                 stencil.names.begin());
  };

  for (auto &record : scratch.records) {
    const std::array<Index, 3> local{local_of(record[0]), local_of(record[1]),
                                     local_of(record[2])};
    const int first =
        tf::fill::hole_canonical_rotation(local[0], local[1], local[2]);
    record = {local[std::size_t(first)], local[std::size_t((first + 1) % 3)],
              local[std::size_t((first + 2) % 3)], record[3]};
  }
  std::sort(scratch.records.begin(), scratch.records.end());
  scratch.records.erase_till_end(
      std::unique(scratch.records.begin(), scratch.records.end()));

  stencil.triangles.clear();
  stencil.from_patch.clear();
  for (const auto &record : scratch.records) {
    stencil.triangles.push_back({record[0], record[1], record[2]});
    stencil.from_patch.push_back(char(record[3]));
  }

  stencil.positions.clear();
  for (auto name : stencil.names) {
    const auto point = name < n_points
                           ? mesh_points[std::size_t(name)]
                                 .template as<double>()
                           : minted[std::size_t(name - n_points)]
                                 .template as<double>();
    stencil.positions.push_back(point);
  }

  stencil.free_of.allocate(stencil.names.size());
  for (std::size_t k = 0; k < stencil.names.size(); ++k)
    stencil.free_of[k] = Index(-1);
  stencil.free_ids.clear();
  for (std::size_t k = 0; k < stencil.names.size(); ++k) {
    const Index name = stencil.names[k];
    if (name < n_points)
      continue;
    if (!std::binary_search(scratch.interior.begin(), scratch.interior.end(),
                            name))
      continue;
    if (std::binary_search(scratch.boundary.begin(), scratch.boundary.end(),
                           name))
      continue;
    stencil.free_of[k] = Index(stencil.free_ids.size());
    stencil.free_ids.push_back(Index(k));
  }
  return complete;
}

} // namespace tf::fill
