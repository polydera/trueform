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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/face_membership_like.hpp"
#include "./hole_metric.hpp"
#include <algorithm>
#include "tbb/parallel_sort.h"
#include <array>
#include <cstddef>

namespace tf::fill {

/// The scratch one seam's statement walks on.
template <typename Index> struct hole_seam_scratch {
  tf::buffer<Index> hosts;
  tf::buffer<std::array<Index, 3>> host_triangles;
  tf::buffer<std::array<Index, 3>> sides;
  tf::buffer<Index> corners;
};

/// The largest angle a patch turns through where it meets the host.
///
/// Every rim subedge is one directed side of a patch triangle and the
/// opposite side of a host triangle across it — the carrying face's own plan
/// — so the seam pairs exactly those sides, and a patch lying flat against
/// its host reads zero. The host triangles are the faces at the patch's
/// original corners, a planned face read as its plan, each rotated onto its
/// smallest corner, and a side's partner is the first of them in that order;
/// a pair either of whose triangles states no plane is not charged. The pass
/// costs the patch's own size once and the host faces at its rim, never a
/// stencil of the whole neighbourhood.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Triangles, typename RealT>
auto hole_patch_seam_angle(
    const tf::polygons<Policy> &polygons,
    const tf::face_membership_like<MembershipPolicy> &fm, Index n_points,
    const Triangles &patch, const tf::buffer<Index> &plan_faces,
    const tf::offset_block_buffer<Index, std::array<Index, 3>> &plan_triangles,
    const tf::buffer<tf::point<RealT, 3>> &minted,
    tf::fill::hole_seam_scratch<Index> &scratch) -> double {
  const auto mesh_faces = polygons.faces();
  const auto mesh_points = polygons.points();
  const auto rotated = [](const std::array<Index, 3> &t) {
    const int first = tf::fill::hole_canonical_rotation(t[0], t[1], t[2]);
    return std::array<Index, 3>{t[std::size_t(first)],
                                t[std::size_t((first + 1) % 3)],
                                t[std::size_t((first + 2) % 3)]};
  };
  const auto position = [&mesh_points, &minted,
                         n_points](Index name) -> tf::point<double, 3> {
    if (name < n_points)
      return mesh_points[std::size_t(name)].template as<double>();
    return minted[std::size_t(name - n_points)].template as<double>();
  };
  const auto normal_of = [&position](const std::array<Index, 3> &t) {
    return tf::fill::hole_oriented_normal(position(t[0]), position(t[1]),
                                          position(t[2]));
  };

  scratch.hosts.clear();
  for (const auto &triangle : patch)
    for (std::size_t corner = 0; corner < 3; ++corner)
      if (triangle[corner] < n_points)
        for (auto face : fm[std::size_t(triangle[corner])])
          scratch.hosts.push_back(Index(face));
  tbb::parallel_sort(scratch.hosts.begin(), scratch.hosts.end());
  scratch.hosts.erase_till_end(
      std::unique(scratch.hosts.begin(), scratch.hosts.end()));

  scratch.host_triangles.clear();
  for (auto face : scratch.hosts) {
    const auto plan =
        std::lower_bound(plan_faces.begin(), plan_faces.end(), face);
    if (plan != plan_faces.end() && *plan == face) {
      for (const auto &triangle :
           plan_triangles[std::size_t(plan - plan_faces.begin())])
        scratch.host_triangles.push_back(rotated(triangle));
      continue;
    }
    const auto corners = mesh_faces[std::size_t(face)];
    if (corners.size() != 3)
      continue;
    scratch.host_triangles.push_back(
        rotated({Index(corners[0]), Index(corners[1]), Index(corners[2])}));
  }
  tbb::parallel_sort(scratch.host_triangles.begin(),
                     scratch.host_triangles.end());
  scratch.host_triangles.erase_till_end(std::unique(
      scratch.host_triangles.begin(), scratch.host_triangles.end()));

  scratch.sides.clear();
  scratch.corners.clear();
  for (std::size_t t = 0; t < scratch.host_triangles.size(); ++t) {
    const auto &corners = scratch.host_triangles[t];
    for (std::size_t side = 0; side < 3; ++side) {
      scratch.sides.push_back(
          {corners[side], corners[(side + 1) % 3], Index(t)});
      scratch.corners.push_back(corners[side]);
    }
  }
  tbb::parallel_sort(scratch.sides.begin(), scratch.sides.end());
  tbb::parallel_sort(scratch.corners.begin(), scratch.corners.end());
  scratch.corners.erase_till_end(
      std::unique(scratch.corners.begin(), scratch.corners.end()));
  const Index last_corner =
      scratch.corners.size() ? scratch.corners.back() : Index(-1);
  const auto on_host = [&scratch, last_corner](Index name) {
    return name <= last_corner &&
           std::binary_search(scratch.corners.begin(), scratch.corners.end(),
                              name);
  };

  double stated = 0.0;
  tf::blocked_reduce(
      patch, stated,
      [&](const auto &block, double &local) {
        for (const auto &stated_triangle : block) {
          const auto triangle =
              rotated({Index(stated_triangle[0]), Index(stated_triangle[1]),
                       Index(stated_triangle[2])});
          bool touches = false;
          for (std::size_t side = 0; side < 3; ++side)
            touches = touches || (on_host(triangle[side]) &&
                                  on_host(triangle[(side + 1) % 3]));
          if (!touches)
            continue;
          const auto normal = normal_of(triangle);
          if (!tf::fill::hole_normal_is_usable(normal))
            continue;
          for (std::size_t side = 0; side < 3; ++side) {
            const std::array<Index, 3> probe{triangle[(side + 1) % 3],
                                             triangle[side], Index(0)};
            const auto found = std::lower_bound(
                scratch.sides.begin(), scratch.sides.end(), probe,
                [](const std::array<Index, 3> &x,
                   const std::array<Index, 3> &y) {
                  return x[0] != y[0] ? x[0] < y[0] : x[1] < y[1];
                });
            if (found == scratch.sides.end() || (*found)[0] != probe[0] ||
                (*found)[1] != probe[1])
              continue;
            const auto host =
                normal_of(scratch.host_triangles[std::size_t((*found)[2])]);
            if (!tf::fill::hole_normal_is_usable(host))
              continue;
            local = std::max(local, tf::fill::hole_dihedral_angle(normal, host));
          }
        }
      },
      [](double local, double &out) { out = std::max(out, local); },
      tf::checked);
  return stated;
}

} // namespace tf::fill
