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
#include "../../core/buffer.hpp"
#include "../../core/closest_point_on_triangle.hpp"
#include "../../core/distance.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../spatial/search.hpp"
#include "../canonical_plane.hpp"
#include "../meta.hpp"
#include "../tag_of_flat_vertex.hpp"
#include "../vertex_converter.hpp"
#include "./placement_tables.hpp"
#include "./point_on_face.hpp"
#include "./pool/wide_to_double.hpp"
#include "./within_band_of_edge_span.hpp"
#include "./within_band_of_point.hpp"
#include "tbb/parallel_sort.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tf::exact::door {

/// The feature of a face a rim vertex lands on: the nearest corner, the
/// nearest open edge span, or the face itself; `on` is a vertex already
/// exactly on the face, which lands nowhere.
enum class rim_target_kind : std::int8_t { corner, edge, face, on };

/// One rim vertex and the one feature of another form it lands on.
template <typename Index> struct rim_target {
  Index flat;
  Index face;
  /// a corner's flat vertex, an edge's slot on the face, `-1` for the face
  Index feature;
  rim_target_kind kind;
};

/// THE LANDING IS ONE FEATURE PER RIM VERTEX. Every rim vertex walks every
/// other form's tree once, pruned by its distance to a node's box, and the
/// nearest face within the band is its target; the nearest point of that
/// face decides the feature, exactly: a corner within the band is the
/// corner, else an open edge span within the band is the edge, else the
/// face. A rim vertex exactly on another form's face already states where
/// it is and is no gap, so it lands nowhere and is counted. The product is
/// ascending by vertex, one row per landing vertex.
///
/// The face test is taken in double on lattice coordinates with a margin
/// for the rounding of both, and may admit a hair beyond the band: the
/// feature is then decided exactly, and a landing is certified by the band
/// when the vertex is placed.
template <typename Index, typename Int, typename RealType, typename ApplyToForm>
auto find_rim_targets(const placement_tables<Index, Int, RealType> &tables,
                      const tf::exact::vertex_converter<Int, RealType, 3>
                          &converter,
                      const ApplyToForm &apply_to_form, Index n_tags,
                      const tf::buffer<Index> &vertex_offsets,
                      const tf::buffer<Index> &face_offsets,
                      const tf::buffer<char> &rim, Int tolerance,
                      tf::buffer<rim_target<Index>> &targets,
                      std::int64_t &rim_vertices, std::int64_t &on_face)
    -> void {
  using T2 = typename tf::exact::meta<Int>::T2;
  using plane_t = tf::exact::canonical_plane<Int>;
  // a lattice point sits half a unit from its source along each axis, so
  // two of them stand at most two units nearer than their sources do
  const auto reach =
      double(converter.coords.deconvert_tolerance(tolerance + Int(4))) *
      (1.0 + 1e-6);
  const auto reach2 = RealType(reach * reach);
  const auto band = double(tolerance);
  const auto &points = tables.points;
  const auto lattice_point = [&points](Index flat) {
    const auto &p = points[std::size_t(flat)];
    return tf::point<double, 3>{double(p[0]), double(p[1]), double(p[2])};
  };
  struct local_t {
    tf::small_vector<tf::point<Int, 3>, 8> corners;
  };
  tf::buffer<rim_target<Index>> out;
  tf::generic_generate(
      tf::make_sequence_range(vertex_offsets[std::size_t(n_tags)]), out,
      [&](Index flat, tf::buffer<rim_target<Index>> &into, local_t &local) {
        if (!rim[std::size_t(flat)])
          return;
        const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
        const auto at = lattice_point(flat);
        const double magnitude = std::fmax(
            std::fabs(at[0]), std::fmax(std::fabs(at[1]), std::fabs(at[2])));
        const double limit = band + 2.0 + magnitude * 0x1p-50;
        double nearest = limit * limit;
        Index nearest_face = Index(-1);
        apply_to_form(tag, [&](const auto &own) {
          const auto point = tf::transformed(
              own.points()[flat - vertex_offsets[std::size_t(tag)]],
              tf::frame_of(own));
          for (Index other = 0; other < n_tags; ++other) {
            if (other == tag)
              continue;
            const auto base = vertex_offsets[std::size_t(other)];
            const auto face_base = face_offsets[std::size_t(other)];
            apply_to_form(other, [&](const auto &form) {
              const auto faces = form.faces();
              tf::search(
                  form,
                  [&point, reach2](const auto &bv) {
                    return tf::distance2(point, bv) <= reach2;
                  },
                  [&](const auto &primitive) {
                    const auto face = faces[primitive.id()];
                    const auto first = lattice_point(base + Index(face[0]));
                    for (std::size_t k = 1; k + 1 < face.size(); ++k) {
                      const std::array<tf::point<double, 3>, 3> triangle{
                          first, lattice_point(base + Index(face[k])),
                          lattice_point(base + Index(face[k + 1]))};
                      const double d2 = tf::distance2(
                          tf::closest_point_on_triangle(triangle, at), at);
                      const auto candidate = face_base + Index(primitive.id());
                      if (d2 < nearest ||
                          (d2 == nearest && candidate < nearest_face)) {
                        nearest = d2;
                        nearest_face = candidate;
                      }
                    }
                  });
            });
          }
        });
        if (nearest_face == Index(-1))
          return;
        const auto &v = points[std::size_t(flat)];
        const auto corners = tables.face_corners(std::size_t(nearest_face));
        // the nearest corner within the band
        {
          bool found = false;
          T2 best(0);
          Index corner = Index(-1);
          for (const auto c : corners) {
            const auto &p = points[std::size_t(c)];
            if (!within_band_of_point<Int>(v, p, tolerance))
              continue;
            T2 length(0);
            for (std::size_t k = 0; k < 3; ++k) {
              const auto delta =
                  typename tf::exact::meta<Int>::T1(v[k]) - p[k];
              length = length + T2(delta) * T2(delta);
            }
            if (!found || length < best || (length == best && c < corner)) {
              found = true;
              best = length;
              corner = c;
            }
          }
          if (found) {
            into.push_back({flat, nearest_face, corner, rim_target_kind::corner});
            return;
          }
        }
        // the nearest open edge span within the band
        {
          bool found = false;
          edge_span_distance<Int> best{};
          Index slot = Index(-1);
          const auto n = corners.size();
          for (std::size_t j = 0; j < n; ++j) {
            edge_span_distance<Int> distance{};
            if (!within_band_of_edge_span<Int>(
                    v, points[std::size_t(corners[j])],
                    points[std::size_t(corners[(j + 1) % n])], tolerance,
                    distance))
              continue;
            if (!found || nearer_edge_span(distance, best)) {
              found = true;
              best = distance;
              slot = Index(j);
            }
          }
          // a vertex exactly on the edge's line is the exact tier's own
          // statement, and lands nowhere
          if (found) {
            into.push_back({flat, nearest_face, slot,
                            best.cross2 == typename tf::exact::meta<Int>::T3(0)
                                ? rim_target_kind::on
                                : rim_target_kind::edge});
            return;
          }
        }
        // the face: a vertex exactly on it states where it is and lands
        // nowhere; one within the band of its plane, the band read against
        // the plane's own length, lands on it
        local.corners.clear();
        for (const auto c : corners)
          local.corners.push_back(points[std::size_t(c)]);
        if (point_on_face(v, local.corners)) {
          into.push_back({flat, nearest_face, Index(-1), rim_target_kind::on});
          return;
        }
        const plane_t &plane = tables.planes[std::size_t(nearest_face)];
        const plane_t none{};
        if (plane == none)
          return;
        const double length = std::sqrt(
            pool::wide_to_double(plane[0]) * pool::wide_to_double(plane[0]) +
            pool::wide_to_double(plane[1]) * pool::wide_to_double(plane[1]) +
            pool::wide_to_double(plane[2]) * pool::wide_to_double(plane[2]));
        const double away =
            std::fabs(pool::wide_to_double(
                tf::exact::orient3d_plane_value<Int>(plane, v))) /
            length;
        if (away > band * (1.0 + 1e-9) + 2.0)
          return;
        into.push_back({flat, nearest_face, Index(-1), rim_target_kind::face});
      },
      local_t{});
  rim_vertices = 0;
  for (const auto r : rim)
    rim_vertices += r ? 1 : 0;
  tbb::parallel_sort(out.begin(), out.end(),
                     [](const rim_target<Index> &x, const rim_target<Index> &y) {
                       return x.flat < y.flat;
                     });
  on_face = 0;
  targets.clear();
  for (const auto &t : out) {
    if (t.kind == rim_target_kind::on)
      ++on_face;
    else
      targets.push_back(t);
  }
}

} // namespace tf::exact::door
