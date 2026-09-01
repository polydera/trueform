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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact_coordinate_converter.hpp"
#include "../../intersect/graph/loop.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../intersect/records/simple_intersections.hpp"
#include "../../topology/cdt_region_mode.hpp"
#include "../../topology/constrained_delaunay_triangulator.hpp"
#include "./emit_iso_face_regions.hpp"
#include "./iso_cut_regions.hpp"
#include "./prepare_iso_face_cut.hpp"
#include "./split_iso_face_chord.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace tf::iso {

/// WAVE. Cut every face the scalar field crossed. The chords of a face are
/// level sets of the field ON that face and never leave it, so the face is the
/// largest independent carrier: one constrained Delaunay build per face, its
/// components the pieces, its position in the record blocks the join the
/// aggregator's walk in input order supplies.
///
/// A face the build declines holds no piece, and `refused` names it: the record
/// blocks are object-ascending and the aggregate walks them in input order, so
/// the list comes out ascending without a sort.
template <typename Int, typename Index, typename Policy, typename RealT,
          std::size_t Dims, typename GetCategory, typename GetOnCut,
          typename GetCreatedCut, typename LabelType>
auto cut_iso_faces(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const GetCategory &get_category, const GetOnCut &get_on_cut,
    const GetCreatedCut &get_created_cut, tf::buffer<LabelType> &bands)
    -> iso_cut_regions<Index, RealT, Dims> {
  iso_cut_regions<Index, RealT, Dims> out;
  bands.clear();
  const auto records = si.intersections();
  if (records.size() == 0)
    return out;

  const auto conv = tf::make_exact_coordinate_converter<Int>(polygons);
  const auto pts = polygons.points();
  const auto faces = polygons.faces();
  const auto field_points = si.intersection_points();
  const Index n_original = static_cast<Index>(pts.size());
  const Index n_field = static_cast<Index>(field_points.size());
  const auto point_of_flat = [&](Index flat) -> tf::exact::pt3<Int> {
    return flat < n_original ? conv(pts[std::size_t(flat)])
                             : conv(field_points[std::size_t(flat - n_original)]);
  };

  struct work_t {
    tf::constrained_delaunay_triangulator<Index, Int, Int> cdt;
    tf::buffer<tf::intersect::graph::loop_node<Index, Int>> loop_work;
    tf::buffer<tf::intersect::graph::vertex<Index>> loop;
    tf::buffer<Index> chain;
    tf::points_buffer<Int, 2> pts2;
    int chain_orientation = 0;
    tf::buffer<Index> cons;
    tf::buffer<Index> rep;
    tf::buffer<Index> mint_ticket;
    tf::buffer<Index> region_offsets;
    tf::buffer<Index> region_order;
    tf::buffer<std::array<Index, 3>> tris;
    tf::buffer<Index> tri_offsets;
    tf::buffer<Index> faces;
    tf::buffer<LabelType> bands;
    tf::buffer<tf::point<RealT, Dims>> mints;
    tf::buffer<Index> refused;
  };

  const auto task = [&](auto &&range, work_t &local) {
    for (const auto &face_records : range) {
      const auto object = face_records[0].object;
      const auto polygon = polygons[object];
      // an n-gon's leading corners may be collinear, so the projection
      // comes from the corners the face's carrier actually stands on
      const auto frame = tf::exact::make_supported_plane_frame<Int>(
          [&](const auto &consider) {
            for (const auto &corner : polygon)
              consider(conv(corner));
          });
      const std::pair<int, int> axes{frame.ax0, frame.ax1};
      const auto face = faces[object];
      prepare_iso_face_cut<Index, Int>(face_records, face, n_original,
                                       field_points, axes, local,
                                       point_of_flat);
      if (split_iso_face_chord<LabelType>(local, object, n_original,
                                          std::size_t(face.size()),
                                          get_category, get_on_cut,
                                          get_created_cut))
        continue;
      if (!local.cdt.build(
              local.pts2.points(),
              tf::make_edges(tf::make_blocked_range<2>(tf::make_range(local.cons))),
              true, tf::cdt_region_mode::components)) {
        local.refused.push_back(object);
        continue;
      }
      emit_iso_face_regions<LabelType, Index, Int>(
          local, object, polygon, axes, conv, n_original, get_category,
          get_on_cut, get_created_cut);
    }
  };

  const auto aggregate = [&](const work_t &local, const tf::none_t &) {
    if (local.refused.size())
      tf::core::append(local.refused, out.refused);
    if (local.tri_offsets.size() == 0)
      return;
    const auto triangle_base =
        static_cast<Index>(out.triangles.data_buffer().size());
    const auto mint_base = static_cast<Index>(out.minted_points.size());
    auto &tris = out.triangles.data_buffer();
    tf::core::append(local.tris, tris);
    if (local.mints.size() != 0) {
      const Index base = n_original + n_field + mint_base;
      for (auto t = std::size_t(triangle_base); t < tris.size(); ++t)
        for (auto &corner : tris[t])
          if (corner < Index(0))
            corner = base + (Index(-1) - corner);
      tf::core::append(local.mints, out.minted_points);
    }
    auto &offsets = out.triangles.offsets_buffer();
    const auto old = offsets.size();
    offsets.reallocate(old + local.tri_offsets.size());
    for (std::size_t i = 0; i < local.tri_offsets.size(); ++i)
      offsets[old + i] = local.tri_offsets[i] + triangle_base;
    tf::core::append(local.faces, out.faces);
    tf::core::append(local.bands, bands);
  };

  tf::blocked_reduce_sequenced_aggregate(records, tf::none, work_t{}, task,
                                         aggregate);
  if (out.triangles.offsets_buffer().size())
    out.triangles.offsets_buffer().push_back(
        static_cast<Index>(out.triangles.data_buffer().size()));
  return out;
}

} // namespace tf::iso
