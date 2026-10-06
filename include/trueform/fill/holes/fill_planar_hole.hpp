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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/polygons.hpp"
#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../../topology/cdt_refine_status.hpp"
#include "../../topology/cdt_refiner.hpp"
#include "../../topology/cdt_region_mode.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_config.hpp"
#include "../hole_fill_status.hpp"
#include "../hole_split.hpp"
#include "./hole_chord_is_forbidden.hpp"
#include "./hole_lattice.hpp"
#include "./hole_rim.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::fill {

/// The scratch one planar rim's build walks on, the refiner included, so a
/// worker reuses one triangulation carrier across the rims it takes.
///
/// The per-output-point lanes are what the tier reads its own product back
/// through: `flat` the identity a corner is emitted as, `carrier` the
/// carrying triangle a boundary split inherits and `-1` for everything else.
template <typename Index, typename Int> struct hole_planar_scratch {
  tf::buffer<tf::point<Int, 3>> lattice;
  tf::buffer<tf::point<Int, 2>> projected;
  tf::blocked_buffer<Index, 2> constraints;
  tf::cdt_refiner<Index, Int> refiner;
  tf::buffer<Index> flat;
  tf::buffer<Index> carrier;
};

/// Fill an exactly planar rim through the constrained Delaunay refiner.
///
/// The rim's own plane is the authority: its sites project into that plane's
/// axes and the refiner runs on whatever lattice they stand on — its own
/// split points are lattice integers like any other, and the dyadic
/// parameter record, not the coordinate, is the split's identity, exactly as
/// the refined boolean path states its shared splits. Originals come back
/// verbatim, a boundary split is materialized on its ORIGINAL 3D edge from
/// the shared parameter — never on the support plane — and an interior
/// Steiner point lifts through the plane it was placed in. The patch's own
/// chords face the same guard a coarse candidate does; the refiner's
/// constrained edges are the rim subedges and are the stitch.
///
/// Returns false where the rim is not planar or the product does not hold,
/// each of which hands the rim on to the table tier.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT, typename Converter>
auto fill_planar_hole(const tf::polygons<Policy> &polygons,
                      const tf::face_membership_like<MembershipPolicy> &fm,
                      const tf::fill::hole_rim<Index, Int, RealT> &rim,
                      const tf::hole_fill_config &config,
                      const Converter &converter, Index n_points,
                      tf::fill::hole_planar_scratch<Index, Int> &scratch,
                      tf::buffer<std::array<Index, 3>> &triangles,
                      tf::buffer<tf::point<RealT, 3>> &minted,
                      tf::buffer<tf::hole_split<Index>> &splits,
                      tf::hole_refine_status &refined) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  using param_t = typename tf::cdt_refiner<Index, Int>::param_t;
  const Index n = Index(rim.size());
  const auto faces = polygons.faces();
  const auto points = polygons.points();

  scratch.lattice.allocate(std::size_t(n));
  for (Index k = 0; k < n; ++k)
    scratch.lattice[std::size_t(k)] =
        tf::fill::hole_unscaled_point<Int>(rim.sites[std::size_t(k)].pt);

  const auto frame = tf::exact::make_supported_plane_frame<Int>(
      [&scratch](const auto &offer) {
        for (const auto &point : scratch.lattice)
          offer(point);
      });
  if (frame.plane_n[0] == T2(0) && frame.plane_n[1] == T2(0) &&
      frame.plane_n[2] == T2(0))
    return false;
  for (const auto &point : scratch.lattice)
    if (frame.plane_n[0] * T2(point[0]) + frame.plane_n[1] * T2(point[1]) +
            frame.plane_n[2] * T2(point[2]) !=
        frame.plane_d)
      return false;

  scratch.projected.allocate(std::size_t(n));
  for (Index k = 0; k < n; ++k)
    scratch.projected[std::size_t(k)] = tf::point<Int, 2>{
        scratch.lattice[std::size_t(k)][std::size_t(frame.ax0)],
        scratch.lattice[std::size_t(k)][std::size_t(frame.ax1)]};
  scratch.constraints.data_buffer().allocate(std::size_t(n) * 2);
  for (Index k = 0; k < n; ++k) {
    scratch.constraints[std::size_t(k)][0] = k;
    scratch.constraints[std::size_t(k)][1] = Index(k + 1 == n ? 0 : k + 1);
  }

  scratch.refiner.always_track_constraint_owners();
  const tf::cdt_refine_config refine{float(config.min_quality), true};
  if (!scratch.refiner.build(tf::make_points(scratch.projected),
                             tf::make_edges(scratch.constraints), refine,
                             tf::cdt_region_mode::nesting))
    return false;

  const Index n_out = Index(scratch.refiner.points().size());
  scratch.flat.allocate(std::size_t(n_out));
  scratch.carrier.allocate(std::size_t(n_out));
  for (Index p = 0; p < n_out; ++p) {
    scratch.flat[std::size_t(p)] = Index(-1);
    scratch.carrier[std::size_t(p)] = Index(-1);
  }
  for (Index k = 0; k < n; ++k) {
    const Index out = scratch.refiner.index_map().f()[std::size_t(k)];
    if (out < 0 || out >= n_out || scratch.flat[std::size_t(out)] != Index(-1))
      return false;
    scratch.flat[std::size_t(out)] = rim.corners[std::size_t(k)];
  }
  Index mints = 0;
  for (Index p = 0; p < n_out; ++p)
    if (scratch.flat[std::size_t(p)] == Index(-1))
      scratch.flat[std::size_t(p)] = Index(n_points + mints++);

  for (Index p = 0; p < n_out; ++p) {
    if (scratch.flat[std::size_t(p)] < n_points)
      continue;
    const auto lifted = tf::exact::lift_plane_point(
        frame, scratch.refiner.points()[std::size_t(p)]);
    minted.push_back(converter.deconvert(lifted));
  }

  const int shift =
      tf::cdt_refiner<Index, Int>::k_crossing_param_bits - tf::hole_split_depth;
  const param_t whole = param_t(1)
                        << tf::cdt_refiner<Index, Int>::k_crossing_param_bits;
  const Index n_faces = scratch.refiner.n_faces();
  for (Index f = 0; f < n_faces; ++f) {
    const auto corners = scratch.refiner.face(f);
    const auto owners = scratch.refiner.face_constraint_owners(f);
    for (int e = 0; e < 3; ++e) {
      const Index input = owners[std::size_t(e)].input_id;
      if (input < 0)
        continue;
      const param_t ends[2] = {owners[std::size_t(e)].t0,
                               owners[std::size_t(e)].t1};
      const Index at[2] = {corners[std::size_t(e)],
                           corners[std::size_t((e + 1) % 3)]};
      for (int side = 0; side < 2; ++side) {
        if (ends[side] == param_t(0) || ends[side] == whole)
          continue;
        if (scratch.carrier[std::size_t(at[side])] != Index(-1))
          continue;
        const param_t scaled = ends[side] >> shift;
        if ((scaled << shift) != ends[side]) {
          minted.clear();
          splits.clear();
          return false;
        }
        const auto &edge = rim.edges[std::size_t(input)];
        const Index from = rim.corners[std::size_t(input)];
        const Index to =
            rim.corners[std::size_t(input + 1 == n ? 0 : input + 1)];
        const int along = int(scaled);
        minted[std::size_t(scratch.flat[std::size_t(at[side])] - n_points)] =
            points[std::size_t(from)] +
            (points[std::size_t(to)] - points[std::size_t(from)]) *
                (RealT(along) / RealT(tf::hole_split_scale));
        scratch.carrier[std::size_t(at[side])] = edge.face;
        splits.push_back(
            {edge.face, edge.v0, edge.v1,
             scratch.flat[std::size_t(at[side])],
             std::uint8_t(edge.t0 == 0 ? along
                                       : tf::hole_split_scale - along)});
      }
    }
  }

  T2 rim_area = T2(0);
  for (Index k = 1; k + 1 < n; ++k)
    rim_area =
        rim_area + tf::exact::orient2d(scratch.projected[0],
                                       scratch.projected[std::size_t(k)],
                                       scratch.projected[std::size_t(k + 1)]);
  const int rim_sign = rim_area > T2(0) ? 1 : rim_area < T2(0) ? -1 : 0;
  if (rim_sign == 0) {
    minted.clear();
    splits.clear();
    return false;
  }

  const auto guarded = [&scratch, &faces, &fm, n_points](Index x, Index y) {
    return tf::fill::hole_chord_is_forbidden(
        faces, fm, n_points, scratch.flat[std::size_t(x)],
        scratch.carrier[std::size_t(x)], scratch.flat[std::size_t(y)],
        scratch.carrier[std::size_t(y)]);
  };

  const auto labels = scratch.refiner.region_labels();
  for (Index f = 0; f < n_faces; ++f) {
    if (labels[std::size_t(f)] % 2 != 1)
      continue;
    const auto corners = scratch.refiner.face(f);
    const auto owners = scratch.refiner.face_constraint_owners(f);
    bool holds = true;
    for (int e = 0; e < 3; ++e)
      if (owners[std::size_t(e)].input_id < 0 &&
          guarded(corners[std::size_t(e)], corners[std::size_t((e + 1) % 3)]))
        holds = false;
    const int sign = tf::exact::orient2d_sign(
        scratch.refiner.points()[std::size_t(corners[0])],
        scratch.refiner.points()[std::size_t(corners[1])],
        scratch.refiner.points()[std::size_t(corners[2])]);
    if (!holds || sign == 0) {
      triangles.clear();
      minted.clear();
      splits.clear();
      return false;
    }
    const bool turn = sign == rim_sign;
    triangles.push_back({scratch.flat[std::size_t(corners[turn ? 2 : 0])],
                         scratch.flat[std::size_t(corners[1])],
                         scratch.flat[std::size_t(corners[turn ? 0 : 2])]});
  }

  switch (scratch.refiner.refine_status()) {
  case tf::cdt_refine_status::floor_met:
    refined = tf::hole_refine_status::floor_met;
    break;
  case tf::cdt_refine_status::stalled:
    refined = tf::hole_refine_status::stalled;
    break;
  case tf::cdt_refine_status::budget_exhausted:
    refined = tf::hole_refine_status::budget_exhausted;
    break;
  }
  return true;
}

} // namespace tf::fill
